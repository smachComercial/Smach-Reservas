"""
Bot de WhatsApp para Los Ciruelos Pádel
Migrado de Telegram + Claude a WhatsApp Cloud API + Mistral.
Conversación: mistral-small-latest | Visión: gemini-2.0-flash | BD: Supabase
"""

import os
import json
import logging
import base64
import httpx
from datetime import datetime, date, timedelta, timezone
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from mistralai import Mistral
from supabase import create_client, Client
from google import genai
from google.genai import types as genai_types

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Variables de entorno ──────────────────────
MISTRAL_API_KEY          = os.environ["MISTRAL_AI_API_KEY"]
WHATSAPP_API_TOKEN       = os.environ["WHATSAPP_API_TOKEN"]
WHATSAPP_PHONE_NUMBER_ID = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
WHATSAPP_VERIFY_TOKEN    = os.environ["WHATSAPP_VERIFY_TOKEN"]
GRAPH_API_VERSION        = os.environ.get("GRAPH_API_VERSION", "v18.0")
SUPABASE_URL             = os.environ["SUPABASE_URL"]
SUPABASE_KEY             = os.environ["SUPABASE_KEY"]
GEMINI_API_KEY           = os.environ["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
mistral = Mistral(api_key=MISTRAL_API_KEY)
gemini = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# ── Zona horaria Argentina (UTC-3 fijo, sin DST) ──
ZONA_ARG = timezone(timedelta(hours=-3))

def ahora_arg() -> datetime:
    return datetime.now(ZONA_ARG)

def hoy_argentina() -> str:
    return ahora_arg().strftime("%Y-%m-%d")

def ahora_str_argentina() -> str:
    return ahora_arg().strftime("%H:%M")

# ── Configuración del club ────────────────────
CANCHAS = {
    1: "Cancha 1 - Interior cemento",
    2: "Cancha 2 - Interior cemento",
    3: "Cancha 3 - Interior cemento",
    4: "Cancha 4 - Exterior blindex y cesped",
}

def generar_horarios():
    slots = []
    m = 8 * 60
    while m < 24 * 60:
        slots.append(f"{m // 60:02d}:{m % 60:02d}")
        m += 90
    return slots

HORARIOS = generar_horarios()

SENA_MONTO         = 10000
SENA_DESTINATARIO  = "Alejandro Santillan"
ADMIN_WA_USERNAME  = "@franv4"
ADMIN_TELEGRAM_ID  = 5377407864  # para notificaciones futuras si se reconecta Telegram

TIMEOUT_ESPERA_MINUTOS = 60

# ── Supabase: reservas ────────────────────────

def reservas_del_dia(fecha: str) -> list:
    try:
        return supabase.table("reservas").select("*").eq("fecha", fecha).execute().data or []
    except Exception as e:
        logger.error(f"Error BD reservas_del_dia: {e}")
        return []

def canchas_libres(fecha: str, hora: str) -> list:
    ocupadas = {
        r["cancha_id"]
        for r in reservas_del_dia(fecha)
        if r["hora"] == hora and r.get("estado") != "cancelada"
    }
    return [c for c in CANCHAS if c not in ocupadas]

def crear_reserva(fecha, hora, cancha_id, nombre, telefono, numero_operacion=None):
    try:
        data = {
            "fecha": fecha,
            "hora": hora,
            "cancha_id": cancha_id,
            "nombre_cliente": nombre,
            "telefono_cliente": telefono,
            "estado": "confirmada",
            "creada_en": ahora_arg().isoformat(),
        }
        if numero_operacion:
            data["numero_operacion"] = numero_operacion
        resp = supabase.table("reservas").insert(data).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.error(f"Error creando reserva: {e}")
        if "23505" in str(e):
            return "DUPLICADO"
        return None

def reservas_por_telefono(telefono: str) -> list:
    try:
        return (
            supabase.table("reservas")
            .select("*")
            .eq("telefono_cliente", telefono)
            .neq("estado", "cancelada")
            .order("fecha")
            .execute()
            .data or []
        )
    except Exception as e:
        logger.error(f"Error reservas_por_telefono: {e}")
        return []

def reserva_por_id(rid: int):
    try:
        data = supabase.table("reservas").select("*").eq("id", rid).execute().data
        return data[0] if data else None
    except Exception:
        return None

def cancelar_reserva_bd(rid: int) -> bool:
    try:
        supabase.table("reservas").update({"estado": "cancelada"}).eq("id", rid).execute()
        return True
    except Exception as e:
        logger.error(f"Error cancelando reserva: {e}")
        return False

def operacion_ya_usada(numero_operacion: str) -> bool:
    try:
        data = (
            supabase.table("reservas")
            .select("id")
            .eq("numero_operacion", numero_operacion)
            .neq("estado", "cancelada")
            .execute()
            .data
        )
        return len(data) > 0
    except Exception as e:
        logger.error(f"Error verificando operacion: {e}")
        return False

def grilla_texto(fecha: str) -> str:
    reservas = [r for r in reservas_del_dia(fecha) if r.get("estado") != "cancelada"]
    idx = {(r["cancha_id"], r["hora"]): r for r in reservas}
    txt = f"Grilla {fecha}:\n"
    txt += f"{'Hora':<7}|{'C1':<7}|{'C2':<7}|{'C3':<7}|{'C4':<7}\n"
    txt += "-" * 37 + "\n"
    for hora in HORARIOS:
        fila = f"{hora:<7}|"
        for cid in range(1, 5):
            r = idx.get((cid, hora))
            if r:
                partes = r["nombre_cliente"].split()
                ini = "".join(p[0].upper() for p in partes[:2])
                fila += f"{ini:<7}|"
            else:
                fila += f"{'libre':<7}|"
        txt += fila + "\n"
    return txt

# ── Supabase: sesiones ────────────────────────
# El campo en BD se llama telegram_user_id pero se usa con wa_id (string).
# No se renombra la columna — compatible con Supabase jsonb/text.

def sesion_get(wa_id: str) -> dict:
    defaults = {
        "esperando_comprobante": False,
        "reserva_pendiente": None,
        "telefono_confirmado": None,
        "nombre_confirmado": None,
        "esperando_desde": None,
        "historial": [],
    }
    try:
        data = (
            supabase.table("sesiones_bot")
            .select("*")
            .eq("telegram_user_id", wa_id)
            .execute()
            .data
        )
        if data:
            s = data[0]
            sesion = {
                "esperando_comprobante": s.get("esperando_comprobante", False),
                "reserva_pendiente": s.get("reserva_pendiente"),
                "telefono_confirmado": s.get("telefono_confirmado"),
                "nombre_confirmado": s.get("nombre_confirmado"),
                "esperando_desde": s.get("esperando_desde"),
                "historial": s.get("historial", []),
            }
            # Auto-resetear si la espera expiró
            if sesion["esperando_comprobante"] and sesion["esperando_desde"]:
                desde = datetime.fromisoformat(sesion["esperando_desde"])
                if desde.tzinfo is None:
                    desde = desde.replace(tzinfo=ZONA_ARG)
                minutos_esperando = (ahora_arg() - desde).total_seconds() / 60
                if minutos_esperando > TIMEOUT_ESPERA_MINUTOS:
                    logger.info(
                        f"Sesión expirada para {wa_id} ({minutos_esperando:.0f} min). Reseteando."
                    )
                    sesion["esperando_comprobante"] = False
                    sesion["reserva_pendiente"] = None
                    sesion["esperando_desde"] = None
                    sesion_set(wa_id, sesion)
            return sesion
    except Exception as e:
        logger.error(f"Error leyendo sesión: {e}")
    return defaults

def sesion_set(wa_id: str, sesion: dict):
    try:
        supabase.table("sesiones_bot").upsert({
            "telegram_user_id": wa_id,
            "esperando_comprobante": sesion.get("esperando_comprobante", False),
            "reserva_pendiente": sesion.get("reserva_pendiente"),
            "telefono_confirmado": sesion.get("telefono_confirmado"),
            "nombre_confirmado": sesion.get("nombre_confirmado"),
            "historial": sesion.get("historial", []),
            "esperando_desde": sesion.get("esperando_desde"),
            "actualizado_en": ahora_arg().isoformat(),
        }).execute()
        logger.info(
            f"sesion_set OK para {wa_id} | esperando_comprobante={sesion.get('esperando_comprobante')}"
        )
    except Exception as e:
        logger.error(f"sesion_set FALLÓ para {wa_id}: {e}")

# ── WhatsApp Cloud API ────────────────────────

GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

def enviar_mensaje_whatsapp(wa_id: str, texto: str):
    """Envía un mensaje de texto a un número de WhatsApp via Graph API."""
    # Normalizar números argentinos: quitar el 9 después del 54 para el sandbox
    if wa_id.startswith("549") and len(wa_id) == 13:
        wa_id = "54" + wa_id[3:]
    url = f"{GRAPH_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": wa_id,
        "type": "text",
        "text": {"body": texto},
    }
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Error enviando mensaje WA a {wa_id}: {e}")

def descargar_imagen_whatsapp(image_id: str) -> bytes | None:
    """
    Descarga una imagen de WhatsApp en dos pasos:
    1. Obtiene la URL de descarga usando el image_id.
    2. Descarga los bytes desde esa URL.
    """
    headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}"}
    try:
        # Paso 1: resolver URL
        meta_url = f"{GRAPH_BASE}/{image_id}"
        r1 = httpx.get(meta_url, headers=headers, timeout=15)
        r1.raise_for_status()
        download_url = r1.json().get("url")
        if not download_url:
            logger.error(f"No se obtuvo URL para image_id={image_id}")
            return None
        # Paso 2: descargar bytes
        r2 = httpx.get(download_url, headers=headers, timeout=30)
        r2.raise_for_status()
        return r2.content
    except Exception as e:
        logger.error(f"Error descargando imagen WA id={image_id}: {e}")
        return None

# ── Verificación de comprobante (Gemini Flash visión) ──

def verificar_comprobante(
    imagen_bytes: bytes,
    media_type: str = "image/jpeg",
    monto_esperado: int = None,
) -> dict:
    """
    Usa gemini-2.0-flash para verificar si la imagen es un comprobante válido.
    Retorna: {"valido": bool, "ilegible": bool, "motivo": str, "numero_operacion": str|null}
    """
    fecha_hoy = ahora_arg().strftime("%d/%m/%Y")
    monto = monto_esperado if monto_esperado is not None else SENA_MONTO
    monto_fmt = f"{monto:,}".replace(",", ".")

    prompt = f"""Analizá esta imagen. Es un comprobante de transferencia bancaria enviado por un cliente para confirmar una reserva de pádel.

Tu tarea es extraer la información visible y verificar si cumple los criterios. Leé con atención todo el texto de la imagen antes de responder.

CRITERIOS A VERIFICAR (los tres deben cumplirse para que sea válido):
1. TIPO: Debe ser un comprobante de transferencia bancaria. Puede ser de Mercado Pago, Naranja X, Brubank, BBVA, Galicia, Santander, Uala, o cualquier banco/billetera argentina.
2. DESTINATARIO: El campo "Para", "Destinatario" o similar debe contener "{SENA_DESTINATARIO}". Aceptá variaciones como: "Roberto Alejandro Santillan", "Alejandro Santillan", "A. Santillan", con o sin tilde en la i. NO es necesario que sea exacto, solo que el apellido "Santillan" esté presente.
3. MONTO: Debe ser exactamente ${monto_fmt} pesos. Puede aparecer como "${monto_fmt}", "$ {monto_fmt}", "{monto}" o similar.
4. FECHA: Debe ser del día de hoy: {fecha_hoy}. Puede estar escrita como "{fecha_hoy}", "hoy", o en formato largo como "jueves 26 de febrero de 2026".

SOBRE "ilegible":
Poné ilegible: true SOLO si la imagen está tan borrosa, oscura o recortada que no podés leer NINGÚN texto. Si podés leer aunque sea parte del texto, poné ilegible: false y evaluá con lo que ves.

NÚMERO DE OPERACIÓN: Buscá cualquier código o número único del pago (puede llamarse "Número de operación", "ID de transacción", "Código", "Referencia", "N° comprobante", etc.) y extraelo. En Mercado Pago suele aparecer al final como "Número de operación de Mercado Pago".

Respondé ÚNICAMENTE con JSON válido, sin texto adicional ni backticks:
{{"valido": true/false, "ilegible": true/false, "motivo": "explicación breve en español de qué cumple o qué falta", "numero_operacion": "el número encontrado o null"}}"""

    try:
        imagen_part = genai_types.Part.from_bytes(data=imagen_bytes, mime_type=media_type)
        response = gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, imagen_part],
        )
        texto = response.text.strip()
        # Limpiar posibles backticks o prefijos de markdown
        texto = texto.replace("```json", "").replace("```", "").strip()
        # Extraer solo el bloque JSON si hay texto extra
        if "{" in texto and "}" in texto:
            inicio = texto.index("{")
            fin = texto.rindex("}") + 1
            texto = texto[inicio:fin]
        resultado = json.loads(texto)
        logger.info(f"Gemini resultado crudo: {resultado}")
        return resultado
    except json.JSONDecodeError as e:
        logger.error(f"Gemini devolvió JSON inválido: {e} | Respuesta: {texto!r}")
        return {
            "valido": False,
            "ilegible": False,
            "motivo": "Error interno al procesar la respuesta. Intentá de nuevo.",
            "numero_operacion": None,
        }
    except Exception as e:
        logger.error(f"Error verificando comprobante con Gemini visión: {e}")
        return {
            "valido": False,
            "ilegible": False,
            "motivo": "Error técnico al conectar con el servicio de verificación.",
            "numero_operacion": None,
        }

# ── Mistral conversacional ────────────────────

def build_system(hoy: str, telefono_conocido: str = None, nombre_conocido: str = None) -> str:
    horarios_str = ", ".join(HORARIOS)
    contexto_telefono = ""
    if telefono_conocido and nombre_conocido:
        contexto_telefono = (
            f"\nDATOS DEL USUARIO: Su nombre es {nombre_conocido} y su telefono confirmado es {telefono_conocido}. "
            f"Si la reserva es a nombre de {nombre_conocido}, usá ese telefono directamente sin pedirlo. "
            "Si la reserva es a nombre de OTRA persona, pedí el telefono de esa persona explicitamente, no uses el guardado."
        )
    elif telefono_conocido:
        contexto_telefono = (
            f"\nDATOS DEL USUARIO: Su telefono confirmado es {telefono_conocido}. "
            "Usalo solo si la reserva es para el mismo usuario; si es para otra persona, pedí su telefono."
        )
    return f"""Sos el asistente virtual del club de padel "Los Ciruelos Padel" en Argentina.
Hablas en espanol rioplatense (vos, tenes, etc.) de forma amigable y natural.

INFORMACION DEL CLUB:
- 4 canchas: Canchas 1, 2 y 3 son interiores de cemento. Cancha 4 es exterior con blindex y cesped.
- Horarios: 08:00 a 00:00, turnos de 1.5 horas
- Turnos validos: {horarios_str}
- Precio del turno: $36.000 (1.5 horas)
- Alquiler de paletas: $4.000 por turno (disponibles en el club)
- Clases disponibles: para consultar contactar al administrador ({ADMIN_WA_USERNAME})
- Instalaciones: buffet y vestuarios disponibles en el club
- Cancha 4 exterior: en caso de lluvia, contactar al administrador ({ADMIN_WA_USERNAME}) para verificar condiciones
- Socios: para asociarse al club contactar al administrador ({ADMIN_WA_USERNAME})
- Hoy es: {hoy} y la hora actual en Argentina es {ahora_str_argentina()}. Usá esta hora para saber qué turnos ya pasaron.
{contexto_telefono}

TU TRABAJO:
Ayudas a los clientes a hacer reservas, consultar disponibilidad, ver sus reservas, cancelar reservas, ver la grilla del dia.

COMO MANEJAR LAS RESERVAS:
Para crear una reserva necesitas recopilar de forma conversacional:
1. Fecha (la calculas si dicen "manana", "el sabado", etc.)
2. Hora (la aproximas al turno valido mas cercano si dicen "a las 9 de la noche" -> 21:30, etc.)
3. Cancha (si no la piden, sugeris la 1 por defecto)
4. Nombre completo
5. Telefono (pedilo siempre explicitamente si no lo tenes confirmado; NUNCA inventes ni asumas un numero)

Cuando tenes todos los datos y el cliente los confirma, emití la accion correspondiente:
- Si es UNA sola reserva: usá preparar_reserva
- Si son DOS O MAS reservas a la vez: usá preparar_multiples_reservas con una lista
En tu mensaje de texto antes de la accion, mostrale el resumen final de los datos confirmados. NO agregues frases como "voy a procesar", "perfecto, ya lo anoto", ni nada después del resumen. El sistema se encarga del resto automáticamente.

ACCIONES DISPONIBLES:
Cuando tengas toda la info necesaria, incluye al FINAL de tu respuesta un bloque JSON entre etiquetas <ACCION></ACCION>.

Para preparar una reserva (cuando tenes TODOS los datos y el cliente los confirmó):
<ACCION>{{"tipo": "preparar_reserva", "fecha": "YYYY-MM-DD", "hora": "HH:MM", "cancha_id": 1, "nombre": "Nombre Apellido", "telefono": "1123456789"}}</ACCION>

Para preparar MULTIPLES reservas a la vez (cuando el cliente confirmó 2 o más turnos juntos):
<ACCION>{{"tipo": "preparar_multiples_reservas", "reservas": [{{"fecha": "YYYY-MM-DD", "hora": "HH:MM", "cancha_id": 1, "nombre": "Nombre Apellido", "telefono": "1123456789"}}, {{"fecha": "YYYY-MM-DD", "hora": "HH:MM", "cancha_id": 1, "nombre": "Nombre Apellido", "telefono": "1123456789"}}]}}</ACCION>

Para consultar disponibilidad:
<ACCION>{{"tipo": "consultar_disponibilidad", "fecha": "YYYY-MM-DD", "hora": "HH:MM"}}</ACCION>

Para ver reservas de un cliente:
<ACCION>{{"tipo": "consultar_reservas", "telefono": "1123456789"}}</ACCION>

Para cancelar una reserva:
<ACCION>{{"tipo": "cancelar_reserva", "reserva_id": 42}}</ACCION>

Para cancelar varias reservas a la vez (cuando el usuario quiere cancelar multiples o "todas"):
<ACCION>{{"tipo": "cancelar_multiples_reservas", "reserva_ids": [42, 43, 44]}}</ACCION>

Para ver la grilla de un dia:
<ACCION>{{"tipo": "ver_grilla", "fecha": "YYYY-MM-DD"}}</ACCION>

Para derivar a una persona real (cuando el usuario lo pide explicitamente o tiene una consulta que no podés resolver):
<ACCION>{{"tipo": "derivar_humano", "motivo": "descripcion breve de la consulta"}}</ACCION>

REGLAS:
- No incluyas el bloque ACCION hasta tener todos los datos necesarios y confirmados
- Si el cliente da una hora aproximada, convierteala al turno valido mas cercano
- IMPORTANTE: Nunca ofrezcas ni aceptes reservas para horarios que ya pasaron hoy. Si el usuario pide un turno de hoy que ya pasó, informale amablemente y pedile que elija otro horario o fecha.
- Cuando recibas un RESULTADO_SISTEMA con TURNO_PASADO, comunicalo de forma amigable y pedile al usuario que elija otro horario.
- Cuando recibas un RESULTADO_SISTEMA, comunicalo de forma amigable
- Mantene las respuestas concisas y naturales
- IMPORTANTE: No uses formato Markdown (sin asteriscos, sin guiones bajos, sin backticks). Texto plano solamente.
- IMPORTANTE: Jamas uses insultos, groserias ni lenguaje ofensivo, aunque el usuario lo haga. Si insultan, respondé con calma y redirigí la conversacion al tema del club.
- IMPORTANTE: Está PROHIBIDO usar las siguientes palabras o cualquier variante de ellas, incluso en tono amistoso o coloquial: boludo, pelotudo, cagaste, la concha, la puta, mierda, culo, forro, hdp, hijo de puta, puto, gil, chabón (en tono despectivo), tarado, mogólico, y cualquier otra grosería o insulto del lunfardo argentino. Usá siempre un español rioplatense amigable pero respetuoso y profesional.
- IMPORTANTE: Si el usuario pide hablar con una persona, un humano, el encargado, o dice que tiene una consulta especial, usa la accion derivar_humano inmediatamente. En tu respuesta de texto (antes del bloque ACCION) decile que vas a avisar al encargado y que puede contactarlo directamente: {ADMIN_WA_USERNAME}.
"""

def ejecutar_accion(accion: dict) -> str:
    tipo = accion.get("tipo")

    if tipo == "derivar_humano":
        motivo = accion.get("motivo", "Sin especificar")
        return f"DERIVAR_HUMANO:{motivo}"

    if tipo == "preparar_reserva":
        hoy_str = hoy_argentina()
        if accion["fecha"] == hoy_str:
            ahora = ahora_arg()
            hora_turno = datetime.strptime(accion["hora"], "%H:%M").replace(
                year=ahora.year, month=ahora.month, day=ahora.day, tzinfo=ZONA_ARG
            )
            if hora_turno <= ahora:
                return (
                    f"TURNO_PASADO: El turno de las {accion['hora']} de hoy ya pasó. "
                    f"Elegí un horario posterior a las {ahora.strftime('%H:%M')}."
                )
        libres = canchas_libres(accion["fecha"], accion["hora"])
        if accion["cancha_id"] not in libres:
            if libres:
                return (
                    f"CANCHA_NO_DISPONIBLE: La cancha {accion['cancha_id']} ya no está libre. "
                    f"Disponibles: {', '.join(CANCHAS[c] for c in libres)}"
                )
            return f"CANCHA_NO_DISPONIBLE: No hay canchas libres para {accion['fecha']} a las {accion['hora']}."
        return "RESERVA_LISTA"

    if tipo == "preparar_multiples_reservas":
        reservas = accion.get("reservas", [])
        if not reservas:
            return "No se especificaron reservas."
        hoy_str = hoy_argentina()
        ahora = ahora_arg()
        for r in reservas:
            if r["fecha"] == hoy_str:
                hora_turno = datetime.strptime(r["hora"], "%H:%M").replace(
                    year=ahora.year, month=ahora.month, day=ahora.day, tzinfo=ZONA_ARG
                )
                if hora_turno <= ahora:
                    return (
                        f"TURNO_PASADO: El turno de las {r['hora']} de hoy ya pasó. "
                        f"Elegí un horario posterior a las {ahora.strftime('%H:%M')}."
                    )
            libres = canchas_libres(r["fecha"], r["hora"])
            if r["cancha_id"] not in libres:
                if libres:
                    return (
                        f"CANCHA_NO_DISPONIBLE: La cancha {r['cancha_id']} para las {r['hora']} no está libre. "
                        f"Disponibles: {', '.join(CANCHAS[c] for c in libres)}"
                    )
                return f"CANCHA_NO_DISPONIBLE: No hay canchas libres para {r['fecha']} a las {r['hora']}."
        return "MULTIPLES_RESERVAS_LISTAS"

    elif tipo == "consultar_disponibilidad":
        fecha, hora = accion["fecha"], accion["hora"]
        libres = canchas_libres(fecha, hora)
        if libres:
            nombres = [CANCHAS[c] for c in libres]
            return f"Canchas libres el {fecha} a las {hora}: {', '.join(nombres)}"
        return f"No hay canchas disponibles el {fecha} a las {hora}."

    elif tipo == "consultar_reservas":
        reservas = reservas_por_telefono(accion["telefono"])
        if not reservas:
            return f"No hay reservas activas para el telefono {accion['telefono']}."
        lineas = [
            f"#{r['id']} - {r['fecha']} {r['hora']} - {CANCHAS[r['cancha_id']]} - {r['nombre_cliente']}"
            for r in reservas
        ]
        return "Reservas encontradas:\n" + "\n".join(lineas)

    elif tipo == "cancelar_reserva":
        rid = accion["reserva_id"]
        reserva = reserva_por_id(rid)
        if not reserva:
            return f"No existe la reserva #{rid}."
        if reserva.get("estado") == "cancelada":
            return f"La reserva #{rid} ya estaba cancelada."
        if cancelar_reserva_bd(rid):
            return (
                f"Reserva #{rid} cancelada. Era: {reserva['fecha']} {reserva['hora']} "
                f"- {CANCHAS[reserva['cancha_id']]} - {reserva['nombre_cliente']}"
            )
        return "ERROR al cancelar."

    elif tipo == "cancelar_multiples_reservas":
        ids = accion.get("reserva_ids", [])
        if not ids:
            return "No se especificaron reservas para cancelar."
        resultados = []
        for rid in ids:
            reserva = reserva_por_id(rid)
            if not reserva:
                resultados.append(f"#{rid}: no existe")
            elif reserva.get("estado") == "cancelada":
                resultados.append(f"#{rid}: ya estaba cancelada")
            elif cancelar_reserva_bd(rid):
                resultados.append(
                    f"#{rid} cancelada ({reserva['fecha']} {reserva['hora']} - {CANCHAS[reserva['cancha_id']]})"
                )
            else:
                resultados.append(f"#{rid}: error al cancelar")
        return "Resultado cancelaciones:\n" + "\n".join(resultados)

    elif tipo == "ver_grilla":
        return grilla_texto(accion["fecha"])

    return "Accion no reconocida."

def llamar_mistral(
    historial: list,
    hoy: str,
    telefono_conocido: str = None,
    nombre_conocido: str = None,
) -> tuple:
    """
    Llama a mistral-small-latest con el historial de conversación.
    Retorna (texto_respuesta, accion_dict | None).
    """
    messages = [{"role": "system", "content": build_system(hoy, telefono_conocido, nombre_conocido)}]
    messages.extend(historial)

    response = mistral.chat.complete(
        model="mistral-small-latest",
        max_tokens=800,
        messages=messages,
    )
    respuesta = response.choices[0].message.content
    accion = None
    texto = respuesta

    if "<ACCION>" in respuesta and "</ACCION>" in respuesta:
        i = respuesta.find("<ACCION>")
        f = respuesta.find("</ACCION>") + len("</ACCION>")
        bloque = respuesta[i:f]
        texto = (respuesta[:i] + respuesta[f:]).strip()
        try:
            json_str = bloque.replace("<ACCION>", "").replace("</ACCION>", "").strip()
            accion = json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parseando accion JSON: {e}")

    return texto, accion

# ── Lógica de mensajes WhatsApp ───────────────

def manejar_mensaje_wa(wa_id: str, texto_usuario: str):
    """Procesa un mensaje de texto entrante de WhatsApp."""
    hoy = hoy_argentina()
    sesion = sesion_get(wa_id)

    # Si estamos esperando comprobante, solo aceptamos cancelación o recordamos
    if sesion["esperando_comprobante"]:
        palabras_cancelar = [
            "cancelar", "cancel", "no quiero", "olvidate", "dejalo",
            "salir", "no importa", "abort",
        ]
        if any(p in texto_usuario.lower() for p in palabras_cancelar):
            sesion["esperando_comprobante"] = False
            sesion["reserva_pendiente"] = None
            sesion["historial"] = []
            sesion_set(wa_id, sesion)
            enviar_mensaje_whatsapp(
                wa_id,
                "Entendido, cancelé el proceso de reserva. Si en algún momento querés intentarlo de nuevo, avisame.",
            )
        else:
            enviar_mensaje_whatsapp(
                wa_id,
                "Estoy esperando el comprobante de la seña para confirmar tu reserva. "
                "Si querés cancelar el proceso, escribí 'cancelar'.",
            )
        return

    historial = sesion["historial"]
    telefono_conocido = sesion["telefono_confirmado"]
    nombre_conocido = sesion.get("nombre_confirmado")
    historial.append({"role": "user", "content": texto_usuario})

    try:
        texto_respuesta, accion = llamar_mistral(historial, hoy, telefono_conocido, nombre_conocido)

        if accion:
            resultado = ejecutar_accion(accion)
            logger.info(f"Accion: {accion.get('tipo')} | Resultado: {resultado[:80]}")

            # Extraer teléfono/nombre de la acción
            telefono_en_accion = accion.get("telefono")
            nombre_en_accion = accion.get("nombre")
            if not telefono_en_accion and accion.get("reservas"):
                telefono_en_accion = accion["reservas"][0].get("telefono")
            if not nombre_en_accion and accion.get("reservas"):
                nombre_en_accion = accion["reservas"][0].get("nombre")
            if telefono_en_accion and not sesion.get("telefono_confirmado"):
                sesion["telefono_confirmado"] = telefono_en_accion
                telefono_conocido = telefono_en_accion
            if nombre_en_accion and not sesion.get("nombre_confirmado"):
                sesion["nombre_confirmado"] = nombre_en_accion
                nombre_conocido = nombre_en_accion

            logger.info(f"Resultado accion: '{resultado}'")

            if resultado == "RESERVA_LISTA":
                sesion["reserva_pendiente"] = [{
                    "fecha": accion["fecha"],
                    "hora": accion["hora"],
                    "cancha_id": accion["cancha_id"],
                    "nombre": accion["nombre"],
                    "telefono": accion["telefono"],
                }]
                sesion["esperando_comprobante"] = True
                sesion["esperando_desde"] = ahora_arg().isoformat()
                sesion["historial"] = []
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(
                    wa_id,
                    "Perfecto! Para confirmar tu reserva necesitás abonar una seña de $10.000 "
                    "por transferencia bancaria a Alejandro Santillan.\n\n"
                    "Una vez que hagas la transferencia, mandame la foto del comprobante y confirmo tu reserva.",
                )

            elif resultado == "MULTIPLES_RESERVAS_LISTAS":
                cantidad = len(accion["reservas"])
                monto_total = cantidad * SENA_MONTO
                monto_fmt = f"{monto_total:,}".replace(",", ".")
                sesion["reserva_pendiente"] = accion["reservas"]
                sesion["esperando_comprobante"] = True
                sesion["esperando_desde"] = ahora_arg().isoformat()
                sesion["historial"] = []
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(
                    wa_id,
                    f"Perfecto! Para confirmar tus {cantidad} reservas necesitás abonar una seña de ${monto_fmt} "
                    f"({cantidad} x $10.000) por transferencia bancaria a Alejandro Santillan.\n\n"
                    f"Podés hacer una sola transferencia de ${monto_fmt} o varias de $10.000 cada una. "
                    f"Mandame la foto de cada comprobante y confirmo tus reservas.",
                )

            elif resultado.startswith("DERIVAR_HUMANO:"):
                motivo = resultado.replace("DERIVAR_HUMANO:", "").strip()
                historial.append({"role": "assistant", "content": texto_respuesta})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                # Notificación al admin (log por ahora; se puede conectar a WA/Telegram futuro)
                logger.warning(
                    f"[DERIVAR_HUMANO] wa_id={wa_id} | telefono={sesion.get('telefono_confirmado')} | motivo={motivo}"
                )
                enviar_mensaje_whatsapp(wa_id, texto_respuesta)

            elif resultado.startswith("CANCHA_NO_DISPONIBLE") or resultado.startswith("TURNO_PASADO"):
                mensaje_sistema = resultado.split(": ", 1)[-1]
                historial.append({"role": "assistant", "content": texto_respuesta})
                historial.append({"role": "user", "content": f"<RESULTADO_SISTEMA>{mensaje_sistema}</RESULTADO_SISTEMA>"})
                texto_final, _ = llamar_mistral(historial, hoy, telefono_conocido, nombre_conocido)
                historial.append({"role": "assistant", "content": texto_final})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(wa_id, texto_final)

            else:
                historial.append({"role": "assistant", "content": texto_respuesta})
                historial.append({"role": "user", "content": f"<RESULTADO_SISTEMA>{resultado}</RESULTADO_SISTEMA>"})
                texto_final, _ = llamar_mistral(historial, hoy, telefono_conocido, nombre_conocido)
                historial.append({"role": "assistant", "content": texto_final})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(wa_id, texto_final)

        else:
            historial.append({"role": "assistant", "content": texto_respuesta})
            sesion["historial"] = historial[-20:]
            sesion_set(wa_id, sesion)
            enviar_mensaje_whatsapp(wa_id, texto_respuesta)

    except Exception as e:
        import traceback
        logger.error(f"Error en manejar_mensaje_wa: {traceback.format_exc()}")
        enviar_mensaje_whatsapp(wa_id, "Hubo un problema tecnico, intenta de nuevo.")


def manejar_foto_wa(wa_id: str, image_id: str, media_type: str = "image/jpeg"):
    """Procesa una imagen enviada por el usuario (comprobante de seña)."""
    sesion = sesion_get(wa_id)

    if not sesion["esperando_comprobante"]:
        enviar_mensaje_whatsapp(
            wa_id,
            "No estoy esperando ningún comprobante en este momento. "
            "Si querés hacer una reserva, escribime los datos.",
        )
        return

    reserva_pendiente = sesion["reserva_pendiente"]
    if not reserva_pendiente:
        sesion["esperando_comprobante"] = False
        sesion_set(wa_id, sesion)
        enviar_mensaje_whatsapp(
            wa_id,
            "Hubo un problema con tu reserva pendiente. Por favor empezá de nuevo.",
        )
        return

    # Normalizar siempre a lista
    if isinstance(reserva_pendiente, dict):
        reserva_pendiente = [reserva_pendiente]

    reservas_pendientes = reserva_pendiente
    cantidad_pendiente = len(reservas_pendientes)
    monto_este_comprobante = SENA_MONTO
    monto_total_restante = cantidad_pendiente * SENA_MONTO

    enviar_mensaje_whatsapp(wa_id, "Recibí el comprobante, lo estoy verificando...")

    try:
        imagen_bytes = descargar_imagen_whatsapp(image_id)
        if not imagen_bytes:
            enviar_mensaje_whatsapp(
                wa_id,
                "No pude descargar la imagen. Por favor intentá mandarla de nuevo.",
            )
            return

        # Verificar contra monto total restante
        resultado = verificar_comprobante(imagen_bytes, media_type, monto_total_restante)

        # Si no es válido con el total y hay más de 1 reserva, intentar con $10.000 parcial
        if not resultado.get("valido") and not resultado.get("ilegible") and cantidad_pendiente > 1:
            resultado_parcial = verificar_comprobante(imagen_bytes, media_type, SENA_MONTO)
            if resultado_parcial.get("valido"):
                resultado = resultado_parcial
                monto_este_comprobante = SENA_MONTO

        logger.info(f"Verificación comprobante wa_id={wa_id}: {resultado}")

        if resultado.get("ilegible"):
            enviar_mensaje_whatsapp(
                wa_id,
                "No pude leer bien la imagen, está borrosa o cortada. "
                "Por favor mandá otra foto más clara del comprobante.",
            )
            return

        if resultado.get("valido"):
            numero_operacion = resultado.get("numero_operacion")

            if numero_operacion and operacion_ya_usada(numero_operacion):
                enviar_mensaje_whatsapp(
                    wa_id,
                    "Este comprobante ya fue utilizado para otra reserva. "
                    "Por favor realizá una nueva transferencia y mandá el comprobante nuevo.",
                )
                return

            if monto_este_comprobante >= monto_total_restante or cantidad_pendiente == 1:
                # Confirmar todas las reservas pendientes
                reservas_creadas = []
                hubo_duplicado = False
                for r in reservas_pendientes:
                    reserva = crear_reserva(
                        r["fecha"], r["hora"], r["cancha_id"],
                        r["nombre"], r["telefono"], numero_operacion,
                    )
                    if reserva == "DUPLICADO":
                        hubo_duplicado = True
                    elif reserva:
                        reservas_creadas.append(reserva)

                if hubo_duplicado:
                    # No resetear sesión — el usuario puede mandar otro comprobante
                    enviar_mensaje_whatsapp(
                        wa_id,
                        "Este comprobante ya fue utilizado para una reserva anterior y no puede reutilizarse.\n\n"
                        "Por favor realizá una nueva transferencia a Alejandro Santillan y mandame el comprobante nuevo.",
                    )
                elif reservas_creadas:
                    sesion["esperando_comprobante"] = False
                    sesion["reserva_pendiente"] = None
                    sesion["telefono_confirmado"] = reservas_pendientes[0]["telefono"]
                    sesion["historial"] = []
                    sesion_set(wa_id, sesion)
                    if len(reservas_creadas) == 1:
                        r_data = reservas_creadas[0]
                        enviar_mensaje_whatsapp(
                            wa_id,
                            f"Comprobante verificado correctamente.\n\n"
                            f"Tu reserva quedo confirmada:\n"
                            f"ID: #{r_data['id']}\n"
                            f"Fecha: {r_data['fecha']}\n"
                            f"Hora: {r_data['hora']}\n"
                            f"Cancha: {CANCHAS[r_data['cancha_id']]}\n"
                            f"Nombre: {r_data['nombre_cliente']}\n\n"
                            f"Nos vemos en la cancha!",
                        )
                    else:
                        lineas = "\n".join(
                            f"#{r['id']} - {r['fecha']} {r['hora']} - {CANCHAS[r['cancha_id']]}"
                            for r in reservas_creadas
                        )
                        enviar_mensaje_whatsapp(
                            wa_id,
                            f"Comprobante verificado correctamente.\n\n"
                            f"Tus {len(reservas_creadas)} reservas quedaron confirmadas:\n"
                            f"{lineas}\n\n"
                            f"Nos vemos en la cancha!",
                        )
                else:
                    # Error técnico — no resetear sesión, el usuario puede reintentar
                    enviar_mensaje_whatsapp(
                        wa_id,
                        "Hubo un problema técnico al guardar tu reserva. "
                        "Por favor intentá mandar el comprobante de nuevo o contactá al club directamente.",
                    )
            else:
                # Pago parcial: confirmar solo la primera reserva de la lista
                r = reservas_pendientes[0]
                reserva = crear_reserva(
                    r["fecha"], r["hora"], r["cancha_id"],
                    r["nombre"], r["telefono"], numero_operacion,
                )

                restantes = reservas_pendientes[1:]
                sesion["reserva_pendiente"] = restantes
                if reserva and reserva != "DUPLICADO":
                    sesion["telefono_confirmado"] = r["telefono"]
                sesion["historial"] = []
                sesion_set(wa_id, sesion)

                cantidad_restante = len(restantes)
                monto_restante = cantidad_restante * SENA_MONTO
                monto_restante_fmt = f"{monto_restante:,}".replace(",", ".")

                if reserva and reserva != "DUPLICADO":
                    enviar_mensaje_whatsapp(
                        wa_id,
                        f"Comprobante verificado. Reserva confirmada:\n"
                        f"ID: #{reserva['id']} - {r['fecha']} {r['hora']} - {CANCHAS[r['cancha_id']]}\n\n"
                        f"Todavía te falta abonar la seña de {cantidad_restante} reserva"
                        f"{'s' if cantidad_restante > 1 else ''} más "
                        f"(${monto_restante_fmt}). Mandame el próximo comprobante.",
                    )
                else:
                    enviar_mensaje_whatsapp(
                        wa_id,
                        "El comprobante es válido pero hubo un problema técnico al guardar la reserva. "
                        "Por favor contactá al club directamente.",
                    )
        else:
            motivo = resultado.get("motivo", "No cumple los requisitos")
            monto_fmt = f"{monto_total_restante:,}".replace(",", ".")
            # Distinguir error técnico de rechazo real por criterios
            if "error técnico" in motivo.lower() or "error interno" in motivo.lower() or "servicio" in motivo.lower():
                enviar_mensaje_whatsapp(
                    wa_id,
                    "Hubo un problema técnico al verificar el comprobante. "
                    "Por favor mandalo de nuevo en unos segundos.",
                )
            else:
                enviar_mensaje_whatsapp(
                    wa_id,
                    f"El comprobante no es válido: {motivo}\n\n"
                    f"Recordá que la seña debe ser de ${monto_fmt} por transferencia bancaria a Alejandro Santillan. "
                    f"Mandame otra foto cuando lo tengas.",
                )

    except Exception as e:
        import traceback
        logger.error(f"Error procesando foto wa_id={wa_id}: {traceback.format_exc()}")
        enviar_mensaje_whatsapp(
            wa_id,
            "Hubo un problema técnico al procesar la imagen. Intentá mandar la foto de nuevo.",
        )

# ── FastAPI Webhook ───────────────────────────

@app.get("/webhook")
async def webhook_verificar(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """Validación del webhook por parte de Meta."""
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verificado por Meta.")
        return PlainTextResponse(hub_challenge)
    logger.warning("Intento de verificación de webhook fallido.")
    raise HTTPException(status_code=403, detail="Verificación fallida")


@app.post("/webhook")
async def webhook_recibir(request: Request):
    """Procesa mensajes entrantes de WhatsApp."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    try:
        entry = body.get("entry", [])
        for e in entry:
            for change in e.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                for msg in messages:
                    wa_id = msg.get("from")
                    msg_type = msg.get("type")

                    if msg_type == "text":
                        texto = msg.get("text", {}).get("body", "").strip()
                        if texto:
                            manejar_mensaje_wa(wa_id, texto)

                    elif msg_type == "image":
                        image_info = msg.get("image", {})
                        image_id = image_info.get("id")
                        mime_type = image_info.get("mime_type", "image/jpeg")
                        if image_id:
                            manejar_foto_wa(wa_id, image_id, mime_type)

                    else:
                        logger.info(f"Tipo de mensaje no soportado: {msg_type} | wa_id={wa_id}")

    except Exception as e:
        import traceback
        logger.error(f"Error procesando webhook: {traceback.format_exc()}")
        # Siempre retornar 200 a Meta para evitar reintentos
    return {"status": "ok"}


# ── Main ──────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
