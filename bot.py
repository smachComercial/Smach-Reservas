"""
Bot de WhatsApp para Los Ciruelos Pádel
Migrado de Telegram + Claude a WhatsApp Cloud API + Mistral.
Conversación: mistral-small-latest | Visión: gemini-2.5-flash-lite | BD: Supabase
"""

import os
import json
import logging
import traceback
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

# ── Deduplicación de webhooks ─────────────────
# Meta puede enviar el mismo evento varias veces. Guardamos los IDs ya procesados
# en un set en memoria. Se limita a 10.000 entradas para no crecer indefinidamente.
_mensajes_procesados: set[str] = set()
_MAX_DEDUP_SIZE = 10_000

def ya_procesado(msg_id: str) -> bool:
    """Retorna True si el mensaje ya fue procesado (duplicado). Si no, lo registra."""
    if msg_id in _mensajes_procesados:
        return True
    if len(_mensajes_procesados) >= _MAX_DEDUP_SIZE:
        _mensajes_procesados.clear()
    _mensajes_procesados.add(msg_id)
    return False

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
    while m <= 22 * 60 + 30:  # Ultimo turno: 22:30
        slots.append(f"{m // 60:02d}:{m % 60:02d}")
        m += 90
    return slots

HORARIOS = generar_horarios()

SENA_MONTO         = 10000
SENA_DESTINATARIO  = "Alejandro Santillan"
ADMIN_WA_NUMBER    = os.environ.get("ADMIN_WA_NUMBER", "5492983448712")

TIMEOUT_ESPERA_MINUTOS = 60

def normalizar_telefono(telefono: str) -> str:
    """
    Normaliza un número de teléfono argentino a 10 dígitos (sin prefijo país, sin 0, sin 15).
    Ejemplos: '02983448712' → '2983448712', '5492983448712' → '2983448712',
              '02983 44-8712' → '2983448712', '1123456789' → '1123456789'
    """
    if not telefono:
        return telefono
    # Dejar solo dígitos
    digits = "".join(c for c in telefono if c.isdigit())
    # Quitar prefijo internacional 54 o 549
    if digits.startswith("549") and len(digits) == 13:
        digits = digits[3:]
    elif digits.startswith("54") and len(digits) == 12:
        digits = digits[2:]
    # Quitar 0 inicial (ej: 02983...)
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    # Quitar 15 después del código de área (ej: 298315...) — poco común pero posible
    # Solo aplicar si tiene 11 dígitos con 15 en posición 4-5
    if len(digits) == 11 and digits[4:6] == "15":
        digits = digits[:4] + digits[6:]
    return digits

# ── Supabase: reservas ────────────────────────

def reservas_del_dia(fecha: str) -> list:
    try:
        return supabase.table("reservas").select("*").eq("fecha", fecha).execute().data or []
    except Exception as e:
        logger.error(f"Error BD reservas_del_dia: {e}")
        return []

def hay_solapamiento(hora_nueva: str, hora_existente: str, duracion_min: int = 90) -> bool:
    """
    Devuelve True si dos turnos de `duracion_min` minutos se superponen.
    Un turno ocupa [inicio, inicio + duracion_min). Dos rangos se solapan si
    inicio_A < fin_B  y  inicio_B < fin_A.
    """
    fmt = "%H:%M"
    ini_a = datetime.strptime(hora_nueva, fmt)
    fin_a = ini_a + timedelta(minutes=duracion_min)
    ini_b = datetime.strptime(hora_existente, fmt)
    fin_b = ini_b + timedelta(minutes=duracion_min)
    return ini_a < fin_b and ini_b < fin_a

def canchas_libres(fecha: str, hora: str, duracion_min: int = 90) -> list:
    """
    Devuelve las canchas sin reservas que se solapen con el turno solicitado.
    Detecta conflictos aunque el turno existente tenga un horario no estándar.
    """
    ocupadas = {
        r["cancha_id"]
        for r in reservas_del_dia(fecha)
        if r.get("estado") != "cancelada"
        and hay_solapamiento(hora, r["hora"], duracion_min)
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
            .gte("fecha", hoy_argentina())
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
    # Unión de horarios estándar + horas reales en BD (para capturar turnos manuales)
    horas_en_bd = {r["hora"] for r in reservas}
    horas_mostrar = sorted(set(HORARIOS) | horas_en_bd)
    idx = {(r["cancha_id"], r["hora"]): r for r in reservas}
    txt = f"Grilla {fecha}:\n"
    txt += f"{'Hora':<7}|{'C1':<7}|{'C2':<7}|{'C3':<7}|{'C4':<7}\n"
    txt += "-" * 37 + "\n"
    for hora in horas_mostrar:
        fila = f"{hora:<7}|"
        for cid in range(1, 5):
            r = idx.get((cid, hora))
            if r:
                partes = r["nombre_cliente"].split()
                ini = "".join(p[0].upper() for p in partes[:2])
                fila += f"{ini:<7}|"
            else:
                # Marcar [sol] si esta franja horaria se superpone con una reserva existente en esa cancha
                ocupada = any(
                    hay_solapamiento(hora, r2["hora"])
                    for r2 in reservas
                    if r2["cancha_id"] == cid
                )
                fila += f"{'[sol]':<7}|" if ocupada else f"{'libre':<7}|"
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
    Usa gemini-2.5-flash-lite para verificar si la imagen es un comprobante válido.
    Retorna: {"valido": bool, "ilegible": bool, "motivo": str, "numero_operacion": str|null}
    """
    fecha_hoy = ahora_arg().strftime("%d/%m/%Y")
    MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = ahora_arg()
    # Solo fecha numérica — el nombre del día en el comprobante puede no coincidir
    # con el día real y no es relevante para validar si el pago es de hoy.
    fecha_hoy_larga = f"{ahora.day} de {MESES_ES[ahora.month - 1]} de {ahora.year}"
    monto = monto_esperado if monto_esperado is not None else SENA_MONTO
    monto_fmt = f"{monto:,}".replace(",", ".")

    prompt = f"""Analizá esta imagen. Es un comprobante de transferencia bancaria enviado por un cliente para confirmar una reserva de pádel.

Tu tarea es extraer la información visible y verificar si cumple los criterios. Leé con atención todo el texto de la imagen antes de responder.

CRITERIOS A VERIFICAR (los tres deben cumplirse para que sea válido):
1. TIPO: Debe ser un comprobante de transferencia bancaria. Puede ser de Mercado Pago, Naranja X, Brubank, BBVA, Galicia, Santander, Uala, o cualquier banco/billetera argentina.
2. DESTINATARIO: El campo "Para", "Destinatario" o similar debe contener "{SENA_DESTINATARIO}". Aceptá variaciones como: "Roberto Alejandro Santillan", "Alejandro Santillan", "A. Santillan", con o sin tilde en la i. NO es necesario que sea exacto, solo que el apellido "Santillan" esté presente.
3. MONTO: Debe ser exactamente ${monto_fmt} pesos. Puede aparecer como "${monto_fmt}", "$ {monto_fmt}", "{monto}" o similar.
4. FECHA: Debe ser del día de hoy. Hoy es {fecha_hoy_larga} ({fecha_hoy}).
   - Ignorá el nombre del día de la semana que aparezca en el comprobante.
   - Verificá que el número de día y el mes coincidan con hoy.
   - Para el año: aceptá tanto el formato completo ({ahora_arg().year}) como el formato corto ({str(ahora_arg().year)[2:]}). 
   - Si el día y mes coinciden pero el año no aparece claramente o está en formato corto, considerá la fecha como válida.
   - Solo rechazá por fecha si el día o el mes claramente no coinciden con hoy.

SOBRE "ilegible":
Poné ilegible: true SOLO si la imagen está tan borrosa, oscura o recortada que no podés leer NINGÚN texto. Si podés leer aunque sea parte del texto, poné ilegible: false y evaluá con lo que ves.

NÚMERO DE OPERACIÓN: Buscá cualquier código o número único del pago (puede llamarse "Número de operación", "ID de transacción", "Código", "Referencia", "N° comprobante", etc.) y extraelo. En Mercado Pago suele aparecer al final como "Número de operación de Mercado Pago".

Respondé ÚNICAMENTE con JSON válido, sin texto adicional ni backticks:
{{"valido": true/false, "ilegible": true/false, "motivo": "explicación breve en español de qué cumple o qué falta", "numero_operacion": "el número encontrado o null"}}"""

    texto = ""
    try:
        imagen_part = genai_types.Part.from_bytes(data=imagen_bytes, mime_type=media_type)
        response = gemini.models.generate_content(
            model="gemini-2.5-flash-lite",
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

IDENTIDAD:
- Sos el asistente de reservas de Los Ciruelos Padel. Si alguien te pregunta qué sos, quién te hizo, qué tecnología usás o cómo funcionás, respondé siempre que sos el asistente virtual del club, sin mencionar ninguna empresa de IA, modelo ni tecnología. Ejemplo: "Soy el asistente virtual de Los Ciruelos Padel, estoy acá para ayudarte con reservas y consultas del club."

INFORMACION DEL CLUB:
- Nombre: Los Ciruelos Padel
- Direccion: Calle 34 num. 1527 e/ 25 y 26, La Plata
- 4 canchas: Canchas 1, 2 y 3 son interiores de cemento. Cancha 4 es exterior con blindex y cesped.
- Horarios: 08:00 a 00:00, turnos de 1.5 horas. El ultimo turno disponible es a las 22:30 (termina a las 00:00). No ofrecer ni aceptar turnos con inicio despues de las 22:30.
- Turnos estandar: {horarios_str} (son referencia orientativa — el sistema también acepta horarios fuera de esta lista si fueron asignados manualmente por el club)
- Si el usuario pide un horario no estandar entre 08:00 y 22:30, usalo tal cual en la accion sin aproximarlo a uno estandar
- Precio del turno: $36.000 (1.5 horas)
- Alquiler de paletas: $4.000 por turno (disponibles en el club)
- Equipamiento: el club vende equipamiento general de padel (paletas, pelotas, indumentaria)
- Clases disponibles: para consultar contactar al administrador al WhatsApp +{ADMIN_WA_NUMBER}
- Instalaciones: buffet y vestuarios disponibles en el club
- Cancha 4 exterior: en caso de lluvia, contactar al administrador al WhatsApp +{ADMIN_WA_NUMBER} para verificar condiciones
- Socios: para asociarse al club contactar al administrador al WhatsApp +{ADMIN_WA_NUMBER}
- Hoy es: {hoy} y la hora actual en Argentina es {ahora_str_argentina()}. Usá esta hora para saber qué turnos ya pasaron.
{contexto_telefono}

TU TRABAJO:
Ayudas a los clientes a hacer reservas, consultar disponibilidad, ver sus reservas, cancelar reservas, ver la grilla del dia.
Tu único tema de conversación es el club de pádel y sus servicios. Si el usuario hace preguntas que no tienen relación con el club, las reservas o el pádel en general, respondé amablemente que solo podés ayudar con temas del club y redirigí la conversación. No respondas preguntas de cultura general, tecnología, política, ni ningún otro tema ajeno al club.

COMO MANEJAR LAS RESERVAS:
Para crear una reserva necesitas recopilar de forma conversacional:
1. Fecha (la calculas si dicen "manana", "el sabado", etc.)
2. Hora (la aproximas al turno valido mas cercano si dicen "a las 9 de la noche" -> 21:30, etc.)
3. Cancha (si no la piden, sugeris la 1 por defecto)
4. Nombre completo
5. Telefono (pedilo siempre explicitamente si no lo tenes confirmado con este mensaje exacto: "¿Cuál es tu número de celular? Escribilo sin el 0 y sin el 15, solo los 10 dígitos."; NUNCA inventes ni asumas un numero)

Cuando tenes todos los datos y el cliente los confirma, emití la accion correspondiente:
- Si es UNA sola reserva: usá preparar_reserva
- Si son DOS O MAS reservas a la vez: usá preparar_multiples_reservas con una lista
En tu mensaje de texto antes de la accion, mostrale el resumen final de los datos confirmados. NO agregues frases como "voy a procesar", "perfecto, ya lo anoto", ni nada después del resumen. El sistema se encarga del resto automáticamente.
- CRÍTICO: JAMÁS escribas el JSON o el bloque <ACCION> en texto visible al usuario. El bloque <ACCION> debe ir EXCLUSIVAMENTE entre las etiquetas <ACCION></ACCION> al final del mensaje, nunca como texto plano ni repetido fuera de ellas. Si el JSON aparece visible en el chat, es un error grave.

IMPORTANTE SOBRE EL PROCESO DE PAGO:
- Vos solo recopilás los datos y emitís la acción preparar_reserva. El sistema luego le pide al cliente el comprobante de la seña.
- NUNCA confirmes la reserva como "lista", "guardada" o "registrada" vos mismo. Solo mostrá el resumen y emití la acción. El sistema se encarga del resto.
- NUNCA le digas al cliente que mande el comprobante — eso lo hace el sistema automáticamente después de tu acción.

ACCIONES DISPONIBLES:
Cuando tengas toda la info necesaria, incluye al FINAL de tu respuesta un bloque JSON entre etiquetas <ACCION></ACCION>.

Para preparar una reserva (cuando tenes TODOS los datos y el cliente los confirmó):
<ACCION>{{"tipo": "preparar_reserva", "fecha": "YYYY-MM-DD", "hora": "HH:MM", "cancha_id": 1, "nombre": "Nombre Apellido", "telefono": "1123456789"}}</ACCION>

Para preparar MULTIPLES reservas a la vez (cuando el cliente confirmó 2 o más turnos juntos):
<ACCION>{{"tipo": "preparar_multiples_reservas", "reservas": [{{"fecha": "YYYY-MM-DD", "hora": "HH:MM", "cancha_id": 1, "nombre": "Nombre Apellido", "telefono": "1123456789"}}, {{"fecha": "YYYY-MM-DD", "hora": "HH:MM", "cancha_id": 1, "nombre": "Nombre Apellido", "telefono": "1123456789"}}]}}</ACCION>

Para consultar disponibilidad de un horario específico:
<ACCION>{{"tipo": "consultar_disponibilidad", "fecha": "YYYY-MM-DD", "hora": "HH:MM"}}</ACCION>

Para consultar qué turnos quedan disponibles en un día (sin hora específica):
<ACCION>{{"tipo": "consultar_disponibilidad", "fecha": "YYYY-MM-DD"}}</ACCION>

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
- IMPORTANTE: Cuando el usuario pida cancelar una reserva, primero consultá sus reservas activas con consultar_reservas y mostráselas. Esperá que el usuario indique explícitamente cuál quiere cancelar antes de emitir la acción cancelar_reserva o cancelar_multiples_reservas. Nunca cancelés sin confirmación explícita del usuario.
- IMPORTANTE — POLITICA DE CANCELACION Y SEÑA: Cuando el usuario cancele una reserva, informale siempre la siguiente política antes de proceder:
  * Si cancela con MENOS de 24 horas de anticipación al turno: pierde la seña abonada.
  * Si cancela con 24 horas o más de anticipación: la seña queda guardada y puede usarse para una futura reserva.
  Tras informar la política, esperá confirmación del usuario antes de proceder con la cancelación.
- IMPORTANTE — CAMBIO DE TURNO: Si el usuario quiere cambiar su turno (de día, hora o cancha), informale que los cambios de turno se gestionan directamente con el administrador y derívalo al WhatsApp +{ADMIN_WA_NUMBER}. Emití también la accion derivar_humano con motivo "cambio de turno". Aclarales además que si avisan el cambio con 24hs o más de anticipación, la seña se les conserva para el nuevo turno.
- CRÍTICO — OBLIGATORIO: Cuando el usuario pida ver disponibilidad, ver sus reservas o ver la grilla, JAMÁS respondas solo con texto. Tu respuesta DEBE contener el bloque <ACCION> correspondiente. Si no incluís el bloque <ACCION>, es un error grave. Están PROHIBIDAS frases como "Voy a buscar...", "Un momento...", "Ahora consulto...", "Déjame verificar..." o cualquier variante. La única respuesta correcta es emitir la <ACCION> de forma inmediata, sin anunciar que vas a hacerlo.
- CRÍTICO — PROHIBIDO: JAMÁS liste turnos o canchas disponibles de memoria o por tu cuenta. Nunca respondas con una lista de horarios sin haber emitido primero la <ACCION> consultar_disponibilidad o ver_grilla y recibido el RESULTADO_SISTEMA. Si el usuario pregunta qué turnos quedan para un día, emití consultar_disponibilidad sin hora y esperá el resultado del sistema antes de responder. Inventar disponibilidad es un error grave.
- Cuando recibas un RESULTADO_SISTEMA con TURNO_PASADO, comunicalo de forma amigable y pedile al usuario que elija otro horario.
- Cuando recibas un RESULTADO_SISTEMA, comunicalo de forma amigable
- Mantene las respuestas concisas y naturales
- IMPORTANTE: No uses formato Markdown (sin asteriscos, sin guiones bajos, sin backticks). Texto plano solamente.
- IMPORTANTE: Jamas uses insultos, groserias ni lenguaje ofensivo, aunque el usuario lo haga. Si insultan, respondé con calma y redirigí la conversacion al tema del club.
- IMPORTANTE: Está PROHIBIDO usar las siguientes palabras o cualquier variante de ellas, incluso en tono amistoso o coloquial: boludo, pelotudo, cagaste, la concha, la puta, mierda, culo, forro, hdp, hijo de puta, puto, gil, chabón (en tono despectivo), tarado, mogólico, y cualquier otra grosería o insulto del lunfardo argentino. Usá siempre un español rioplatense amigable pero respetuoso y profesional.
- CRITICO — OBLIGATORIO: Si el usuario pide hablar con una persona, un humano, el encargado, el administrador, un representante, o dice que tiene una consulta especial que no podés resolver, SIEMPRE debés emitir la accion derivar_humano. NUNCA respondas solo con texto dando el número sin emitir la accion. El bloque <ACCION> es OBLIGATORIO en estos casos. En tu respuesta de texto (antes del bloque ACCION) decile que puede contactar al administrador directamente por WhatsApp al +{ADMIN_WA_NUMBER}.
"""

def ejecutar_accion(accion: dict, telefono_confirmado: str = None) -> str:
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
        fecha = accion["fecha"]
        hora = accion.get("hora")
        if not hora:
            # Modo "qué turnos quedan": devolver todos los slots con al menos una cancha libre
            ahora = ahora_arg()
            hoy_str = hoy_argentina()
            turnos_libres = []
            for slot in HORARIOS:
                # Saltar turnos pasados si es hoy
                if fecha == hoy_str:
                    hora_slot = datetime.strptime(slot, "%H:%M").replace(
                        year=ahora.year, month=ahora.month, day=ahora.day, tzinfo=ZONA_ARG
                    )
                    if hora_slot <= ahora:
                        continue
                libres = canchas_libres(fecha, slot)
                if libres:
                    turnos_libres.append(slot)
            if turnos_libres:
                return f"Turnos con canchas disponibles el {fecha}: {', '.join(turnos_libres)}"
            return f"No hay turnos disponibles el {fecha}."
        # Modo hora específica
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
        if not telefono_confirmado:
            return "CANCELACION_DENEGADA: No tengo tu número de teléfono confirmado. Por favor consultá tus reservas primero indicando tu número."
        if normalizar_telefono(reserva.get("telefono_cliente", "")) != normalizar_telefono(telefono_confirmado):
            return f"CANCELACION_DENEGADA: La reserva #{rid} no pertenece a tu número de teléfono."
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
        if not telefono_confirmado:
            return "CANCELACION_DENEGADA: No tengo tu número de teléfono confirmado. Por favor consultá tus reservas primero indicando tu número."
        resultados = []
        for rid in ids:
            reserva = reserva_por_id(rid)
            if not reserva:
                resultados.append(f"#{rid}: no existe")
            elif reserva.get("estado") == "cancelada":
                resultados.append(f"#{rid}: ya estaba cancelada")
            elif normalizar_telefono(reserva.get("telefono_cliente", "")) != normalizar_telefono(telefono_confirmado):
                resultados.append(f"#{rid}: no pertenece a tu número")
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

    # Evita error 400 de Mistral si el texto queda vacío (solo había <ACCION>)
    if not texto:
        texto = "."

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
            resultado = ejecutar_accion(accion, sesion.get("telefono_confirmado"))
            logger.info(f"Accion: {accion.get('tipo')} | Resultado: {resultado[:80]}")

            # Extraer teléfono/nombre de la acción
            telefono_en_accion = accion.get("telefono")
            nombre_en_accion = accion.get("nombre")
            if not telefono_en_accion and accion.get("reservas"):
                telefono_en_accion = accion["reservas"][0].get("telefono")
            if not nombre_en_accion and accion.get("reservas"):
                nombre_en_accion = accion["reservas"][0].get("nombre")
            if telefono_en_accion and not sesion.get("telefono_confirmado"):
                sesion["telefono_confirmado"] = normalizar_telefono(telefono_en_accion)
                telefono_conocido = sesion["telefono_confirmado"]
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
                    "Alias: lcpadel.mp\n"
                    "Cuenta: Mercado Pago\n\n"
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
                    "Alias: lcpadel.mp\n"
                    "Cuenta: Mercado Pago\n\n"
                    f"Podés hacer una sola transferencia de ${monto_fmt} o varias de $10.000 cada una. "
                    f"Mandame la foto de cada comprobante y confirmo tus reservas.",
                )

            elif resultado.startswith("DERIVAR_HUMANO:"):
                motivo = resultado.replace("DERIVAR_HUMANO:", "").strip()
                historial.append({"role": "assistant", "content": texto_respuesta})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                logger.warning(
                    f"[DERIVAR_HUMANO] wa_id={wa_id} | telefono={sesion.get('telefono_confirmado')} | motivo={motivo}"
                )
                # Notificación al admin por WhatsApp
                telefono_usuario = sesion.get("telefono_confirmado") or wa_id
                nombre_usuario = sesion.get("nombre_confirmado") or "desconocido"
                notif = (
                    f"[Los Ciruelos Bot] Un cliente quiere hablar con una persona.\n\n"
                    f"Nombre: {nombre_usuario}\n"
                    f"Teléfono/wa_id: {telefono_usuario}\n"
                    f"Motivo: {motivo}"
                )
                enviar_mensaje_whatsapp(ADMIN_WA_NUMBER, notif)
                enviar_mensaje_whatsapp(wa_id, texto_respuesta)

            elif resultado.startswith("CANCELACION_DENEGADA:"):
                mensaje = resultado.split(": ", 1)[-1]
                historial.append({"role": "assistant", "content": mensaje})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(wa_id, mensaje)

            elif resultado.startswith("CANCHA_NO_DISPONIBLE") or resultado.startswith("TURNO_PASADO"):
                mensaje_sistema = resultado.split(": ", 1)[-1]
                historial.append({"role": "assistant", "content": texto_respuesta})
                historial.append({"role": "user", "content": f"<RESULTADO_SISTEMA>{mensaje_sistema}</RESULTADO_SISTEMA>"})
                # Limpiar historial largo para evitar que Mistral reutilice horarios viejos
                historial_limpio = historial[-6:]
                texto_final, _ = llamar_mistral(historial_limpio, hoy, telefono_conocido, nombre_conocido)
                historial.append({"role": "assistant", "content": texto_final})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(wa_id, texto_final)

            else:
                historial.append({"role": "assistant", "content": texto_respuesta})
                historial.append({"role": "user", "content": f"<RESULTADO_SISTEMA>{resultado}</RESULTADO_SISTEMA>"})
                texto_final, accion_final = llamar_mistral(historial, hoy, telefono_conocido, nombre_conocido)
                # Si Mistral volvió a emitir una ACCION en la segunda llamada, procesarla correctamente
                # en lugar de mandar el texto (que podría contener JSON crudo)
                if accion_final:
                    logger.warning(f"Mistral emitió ACCION inesperada en segunda llamada: {accion_final.get('tipo')}. Reprocesando.")
                    resultado2 = ejecutar_accion(accion_final, sesion.get("telefono_confirmado"))
                    logger.info(f"Resultado segunda accion: '{resultado2}'")
                    if resultado2 == "RESERVA_LISTA":
                        sesion["reserva_pendiente"] = [{
                            "fecha": accion_final["fecha"],
                            "hora": accion_final["hora"],
                            "cancha_id": accion_final["cancha_id"],
                            "nombre": accion_final["nombre"],
                            "telefono": accion_final["telefono"],
                        }]
                        sesion["esperando_comprobante"] = True
                        sesion["esperando_desde"] = ahora_arg().isoformat()
                        sesion["historial"] = []
                        sesion_set(wa_id, sesion)
                        enviar_mensaje_whatsapp(
                            wa_id,
                            "Perfecto! Para confirmar tu reserva necesitás abonar una seña de $10.000 "
                            "por transferencia bancaria a Alejandro Santillan.\n\n"
                            "Alias: lcpadel.mp\n"
                            "Cuenta: Mercado Pago\n\n"
                            "Una vez que hagas la transferencia, mandame la foto del comprobante y confirmo tu reserva.",
                        )
                    elif resultado2 == "MULTIPLES_RESERVAS_LISTAS":
                        cantidad = len(accion_final["reservas"])
                        monto_total = cantidad * SENA_MONTO
                        monto_fmt = f"{monto_total:,}".replace(",", ".")
                        sesion["reserva_pendiente"] = accion_final["reservas"]
                        sesion["esperando_comprobante"] = True
                        sesion["esperando_desde"] = ahora_arg().isoformat()
                        sesion["historial"] = []
                        sesion_set(wa_id, sesion)
                        enviar_mensaje_whatsapp(
                            wa_id,
                            f"Perfecto! Para confirmar tus {cantidad} reservas necesitás abonar una seña de ${monto_fmt} "
                            f"({cantidad} x $10.000) por transferencia bancaria a Alejandro Santillan.\n\n"
                            "Alias: lcpadel.mp\n"
                            "Cuenta: Mercado Pago\n\n"
                            f"Podés hacer una sola transferencia de ${monto_fmt} o varias de $10.000 cada una. "
                            f"Mandame la foto de cada comprobante y confirmo tus reservas.",
                        )
                    else:
                        # Para cualquier otro resultado, seguir el flujo normal.
                        # Se descarta cualquier ACCION que Mistral emita en esta llamada —
                        # solo queremos texto natural para comunicar el resultado al usuario.
                        historial.append({"role": "assistant", "content": texto_final})
                        historial.append({"role": "user", "content": f"<RESULTADO_SISTEMA>{resultado2}</RESULTADO_SISTEMA>"})
                        texto_final2, accion_descartada = llamar_mistral(historial, hoy, telefono_conocido, nombre_conocido)
                        if accion_descartada:
                            logger.warning(f"Mistral emitió ACCION inesperada en llamada de traducción (ignorada): {accion_descartada.get('tipo')}")
                        historial.append({"role": "assistant", "content": texto_final2})
                        sesion["historial"] = historial[-20:]
                        sesion_set(wa_id, sesion)
                        enviar_mensaje_whatsapp(wa_id, texto_final2)
                else:
                    historial.append({"role": "assistant", "content": texto_final})
                    sesion["historial"] = historial[-20:]
                    sesion_set(wa_id, sesion)
                    enviar_mensaje_whatsapp(wa_id, texto_final)

        else:
            # Retry: si Mistral no emitió ACCION cuando debería haberlo hecho
            PALABRAS_CONSULTA = [
                "disponib", "grilla",
                "mis reservas", "mis turnos", "tengo reserva",
            ]
            texto_lower = texto_usuario.lower()
            es_consulta = any(p in texto_lower for p in PALABRAS_CONSULTA)

            if es_consulta:
                logger.warning(f"Mistral no emitió ACCION para consulta de '{texto_usuario}'. Ejecutando retry.")
                historial_retry = historial[:-1]  # sin el último mensaje del usuario
                historial_retry.append({
                    "role": "user",
                    "content": (
                        texto_usuario +
                        "\n\n[SISTEMA: Tu respuesta anterior no incluyó el bloque <ACCION>. "
                        "Esto es un error. Debés emitir la <ACCION> correspondiente ahora mismo, "
                        "sin texto previo ni frases de transición.]"
                    )
                })
                texto_retry, accion_retry = llamar_mistral(historial_retry, hoy, telefono_conocido, nombre_conocido)

                if accion_retry:
                    logger.info(f"Retry exitoso. Accion obtenida: {accion_retry.get('tipo')}")
                    resultado_retry = ejecutar_accion(accion_retry, sesion.get("telefono_confirmado"))
                    historial.append({"role": "assistant", "content": texto_retry})
                    historial.append({"role": "user", "content": f"<RESULTADO_SISTEMA>{resultado_retry}</RESULTADO_SISTEMA>"})
                    texto_final, accion_final = llamar_mistral(historial, hoy, telefono_conocido, nombre_conocido)
                    # Si Mistral volvió a emitir una ACCION, procesarla en lugar de mandar JSON crudo
                    if accion_final:
                        logger.warning(f"Mistral emitió ACCION inesperada en llamada post-retry: {accion_final.get('tipo')}. Reprocesando.")
                        resultado_final = ejecutar_accion(accion_final, sesion.get("telefono_confirmado"))
                        if resultado_final == "RESERVA_LISTA":
                            sesion["reserva_pendiente"] = [{
                                "fecha": accion_final["fecha"],
                                "hora": accion_final["hora"],
                                "cancha_id": accion_final["cancha_id"],
                                "nombre": accion_final["nombre"],
                                "telefono": accion_final["telefono"],
                            }]
                            sesion["esperando_comprobante"] = True
                            sesion["esperando_desde"] = ahora_arg().isoformat()
                            sesion["historial"] = []
                            sesion_set(wa_id, sesion)
                            enviar_mensaje_whatsapp(
                                wa_id,
                                "Perfecto! Para confirmar tu reserva necesitás abonar una seña de $10.000 "
                                "por transferencia bancaria a Alejandro Santillan.\n\n"
                                "Alias: lcpadel.mp\n"
                                "Cuenta: Mercado Pago\n\n"
                                "Una vez que hagas la transferencia, mandame la foto del comprobante y confirmo tu reserva.",
                            )
                        elif resultado_final == "MULTIPLES_RESERVAS_LISTAS":
                            cantidad = len(accion_final["reservas"])
                            monto_total = cantidad * SENA_MONTO
                            monto_fmt = f"{monto_total:,}".replace(",", ".")
                            sesion["reserva_pendiente"] = accion_final["reservas"]
                            sesion["esperando_comprobante"] = True
                            sesion["esperando_desde"] = ahora_arg().isoformat()
                            sesion["historial"] = []
                            sesion_set(wa_id, sesion)
                            enviar_mensaje_whatsapp(
                                wa_id,
                                f"Perfecto! Para confirmar tus {cantidad} reservas necesitás abonar una seña de ${monto_fmt} "
                                f"({cantidad} x $10.000) por transferencia bancaria a Alejandro Santillan.\n\n"
                                "Alias: lcpadel.mp\n"
                                "Cuenta: Mercado Pago\n\n"
                                f"Podés hacer una sola transferencia de ${monto_fmt} o varias de $10.000 cada una. "
                                f"Mandame la foto de cada comprobante y confirmo tus reservas.",
                            )
                        else:
                            historial.append({"role": "assistant", "content": texto_final})
                            sesion["historial"] = historial[-20:]
                            sesion_set(wa_id, sesion)
                            enviar_mensaje_whatsapp(wa_id, texto_final)
                    else:
                        historial.append({"role": "assistant", "content": texto_final})
                        sesion["historial"] = historial[-20:]
                        sesion_set(wa_id, sesion)
                        enviar_mensaje_whatsapp(wa_id, texto_final)
                else:
                    logger.warning(f"Retry fallido. Mistral tampoco emitió ACCION en segundo intento.")
                    historial.append({"role": "assistant", "content": texto_respuesta})
                    sesion["historial"] = historial[-20:]
                    sesion_set(wa_id, sesion)
                    enviar_mensaje_whatsapp(wa_id, texto_respuesta)
            else:
                historial.append({"role": "assistant", "content": texto_respuesta})
                sesion["historial"] = historial[-20:]
                sesion_set(wa_id, sesion)
                enviar_mensaje_whatsapp(wa_id, texto_respuesta)

    except Exception as e:
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
                    msg_id = msg.get("id")
                    msg_type = msg.get("type")

                    if msg_id and ya_procesado(msg_id):
                        logger.info(f"Mensaje duplicado ignorado: msg_id={msg_id} wa_id={wa_id}")
                        continue

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

                    elif msg_type == "document":
                        doc_info = msg.get("document", {})
                        doc_id = doc_info.get("id")
                        mime_type = doc_info.get("mime_type", "image/jpeg")
                        if doc_id:
                            manejar_foto_wa(wa_id, doc_id, mime_type)

                    else:
                        logger.info(f"Tipo de mensaje no soportado: {msg_type} | wa_id={wa_id}")

    except Exception as e:
        logger.error(f"Error procesando webhook: {traceback.format_exc()}")
        # Siempre retornar 200 a Meta para evitar reintentos
    return {"status": "ok"}


# ── Main ──────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
