import os
import sqlite3
import json
import requests
import streamlit as st
import time
import chromadb
from google import genai

from google.genai import types
from google.genai.errors import ServerError, APIError

def obtener_embedding(texto: str) -> list[float]:
    """Genera el vector de embeddings usando el modelo actual activo."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texto
    )
    # Extraer la lista de valores (3072 dimensiones)
    if hasattr(response, 'embeddings') and response.embeddings:
        return response.embeddings[0].values
    elif hasattr(response, 'embedding') and response.embedding:
        return response.embedding.values
    return []
# ---------------------------------------------------------------------------
# 1. Configuración Inicial y Secrets
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Agente IA Personal - RAG + GitHub", page_icon="🤖")
st.title("🤖 Asistente Virtual Personal con RAG (CV + GitHub) & SQL")

# Obtener API Keys y configuraciones (seguro si no existe secrets.toml)
try:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    github_user = st.secrets.get("GITHUB_USERNAME") or os.getenv("GITHUB_USERNAME", "juandivison")
    github_token = st.secrets.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")
    github_user = os.getenv("GITHUB_USERNAME", "juandivison")
    github_token = os.getenv("GITHUB_TOKEN")

if not api_key:
    st.error("⚠️ Configura la variable GEMINI_API_KEY en los secretos de Streamlit o variables de entorno.")
    st.stop()

# Inicializar cliente normal (sin forzar v1, dejando que el SDK decida)
client = genai.Client(api_key=api_key)
# ---------------------------------------------------------------------------
# 2. Función para Obtener Repositorios de GitHub
# ---------------------------------------------------------------------------
def obtener_repositorios_github(usuario: str, token: str = None) -> list[str]:
    """Obtiene y formatea los proyectos públicos y privados de un usuario en GitHub usando token."""
    if not usuario or usuario == "tu_usuario_github":
        return []
    
    # Endpoint de la API de GitHub para los repositorios del usuario autenticado
    url = "https://api.github.com/user/repos?per_page=100&sort=updated&type=all"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    # Si hay token, lo incluimos para poder ver repositorios privados
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        # Fallback a repositorios públicos si no hay token
        url = f"https://api.github.com/users/{usuario}/repos?per_page=100&sort=updated"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            repos = response.json()
            documentos_repos = []
            for repo in repos:
                nombre = repo.get("name", "")
                descripcion = repo.get("description") or "Sin descripción"
                lenguaje = repo.get("language") or "No especificado"
                es_privado = "Privado" if repo.get("private") else "Público"
                topics = ", ".join(repo.get("topics", []))
                
                texto = f"Proyecto de GitHub [{es_privado}] '{nombre}': {descripcion}. Lenguaje principal: {lenguaje}."
                if topics:
                    texto += f" Tecnologías/Temas: {topics}."
                
                documentos_repos.append(texto)
            return documentos_repos
        else:
            st.warning(f"No se pudieron obtener repositorios de GitHub (Código HTTP {response.status_code}).")
            return []
    except Exception as e:
        st.warning(f"Error al conectar con la API de GitHub: {e}")
        return []
# ---------------------------------------------------------------------------
# 3. Inicialización y Poblado de la BD Vectorial (RAG)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_rag_vector_db(usuario_gh: str):
    """Inicializa ChromaDB e indexa tanto el CV como los proyectos de GitHub."""
    chroma_client = chromadb.PersistentClient(path="./cv_vector_db")
    collection = chroma_client.get_or_create_collection(name="cv_knowledge_base")

    # Si la base de datos está vacía, proceder a indexar
    if collection.count() == 0:
        # 1. Fragmentos del CV
        
        datos_cv = cargar_y_fragmentar_cv("cv.md")
        datos_linkedin = cargar_y_fragmentar_cv("linkedin.md")

        # 2. Fragmentos de GitHub
        datos_github = obtener_repositorios_github(usuario_gh, github_token)
        
        # Unir todos los documentos a indexar
        documentos_totales = datos_cv + datos_linkedin + datos_github

        # Generar embeddings e indexar en ChromaDB
        for i, texto in enumerate(documentos_totales):
            embedding_vector = obtener_embedding(texto)
            collection.add(
                documents=[texto],
                embeddings=[embedding_vector],
                ids=[f"rag_doc_{i}"]
            )
            
    return collection

# Llamada a la función cacheada limpia de elementos UI
collection = init_rag_vector_db(github_user)
# ---------------------------------------------------------------------------
# 4. Base de Datos SQL (Herramienta Estructurada)
# ---------------------------------------------------------------------------
def init_sql_db():
    conn = sqlite3.connect("agente_stats.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS estadisticas_proyectos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            categoria TEXT,
            cantidad_proyectos INTEGER,
            nivel_experiencia TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM estadisticas_proyectos")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO estadisticas_proyectos (categoria, cantidad_proyectos, nivel_experiencia)
            VALUES (?, ?, ?)
        """, [
            ("ERP & POS (Delphi / Firebird)", 12, "Senior / Arquitecto"),
            ("Servicios Web & APIs (.NET / Node.js)", 8, "Senior"),
            ("Facturación Electrónica e-CF", 5, "Especialista"),
            ("Capacitación Técnica / Docencia", 15, "Facilitador / Tutor")
        ])
        conn.commit()
    conn.close()

init_sql_db()

def cargar_y_fragmentar_cv(ruta_archivo: str = "cv.md", tamano_bloque: int = 500) -> list[str]:
    """Lee el CV de un archivo de texto o Markdown y lo divide en párrafos/bloques."""
    if not os.path.exists(ruta_archivo):
        return []
    
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    # Dividir por párrafos o líneas dobles
    parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
    return parrafos


# ---------------------------------------------------------------------------
# 5. Funciones del Agente RAG
# ---------------------------------------------------------------------------
def buscar_en_rag(pregunta: str) -> str:
    """Busca los fragmentos más relevantes (CV + GitHub) para la consulta."""
    vector_pregunta = obtener_embedding(pregunta)
    resultados = collection.query(query_embeddings=[vector_pregunta], n_results=4)
    docs = resultados.get('documents', [[]])[0]
    return "\n".join(docs) if docs else "No se encontraron fragmentos relevantes."
def ejecutar_consulta_sql(query: str) -> str:
    try:
        conn = sqlite3.connect("agente_stats.db")
        cursor = conn.cursor()
        cursor.execute(query)
        filas = cursor.fetchall()
        columnas = [desc[0] for desc in cursor.description]
        conn.close()
        return json.dumps([dict(zip(columnas, fila)) for fila in filas], ensure_ascii=False)
    except Exception as e:
        return f"Error SQL: {e}"

def generar_con_reintento(prompt: str, max_reintentos: int = 3):
    """Intenta generar contenido con el modelo de IA.
    Si el servidor está ocupado (503), espera y reintenta hasta max_reintentos veces.
    """
    for intento in range(max_reintentos):
        try:
            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
        except ServerError as e:
            # Si es el último intento, re-lanzamos la excepción
            if intento == max_reintentos - 1:
                raise e
            # Esperar 2 segundos con backoff exponencial breve antes del siguiente intento
            time.sleep(2 * (intento + 1))


def responder_usuario(pregunta_usuario: str) -> str:
    try:
        contexto_rag = buscar_en_rag(pregunta_usuario)
        
        contexto_sql = ""
        if any(p in pregunta_usuario.lower() for p in ["cuantos", "cuántos", "cantidad", "estadística", "proyectos", "experiencia"]):
            contexto_sql = ejecutar_consulta_sql("SELECT * FROM estadisticas_proyectos")

        prompt = f"""
        Eres el asistente personal interactivo de Juan Ramón Divisón. 
        Responde las preguntas del usuario basándote en la información recuperada de tu base de conocimientos RAG (que combina el CV y proyectos de GitHub) y herramientas SQL.

        [INFORMACIÓN RECUPERADA DE LA DB RAG (CV + GitHub)]:
        {contexto_rag}

        [ESTADÍSTICAS CONSULTADAS DE LA DB SQL]:
        {contexto_sql}

        Pregunta del usuario: {pregunta_usuario}
        """

        # Llamada con reintento automático
        response = generar_con_reintento(prompt)
        return response.text

    except ServerError:
        return "⚠️ El servicio de inteligencia artificial está experimentando una alta demanda en este momento. Por favor, intenta realizar tu pregunta nuevamente en unos segundos."
    except APIError:
        return "⚠️ Ocurrió un inconveniente temporal de comunicación con los servidores de la IA. Inténtalo de nuevo más tarde."
    except Exception as e:
        return f"⚠️ Ocurrió un error inesperado al procesar tu solicitud: {e}"
# ---------------------------------------------------------------------------
# 6. Interfaz de Usuario con Streamlit
# ---------------------------------------------------------------------------
# Sidebar para configuración opcional de GitHub
with st.sidebar:
    st.header("Configuración")
    input_gh = st.text_input("Usuario de GitHub:", value=github_user)
    if input_gh != github_user:
        st.info("Para aplicar un nuevo usuario de GitHub, elimina la carpeta ./cv_vector_db y reinicia.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Haz una pregunta sobre mi CV o mis repositorios de GitHub..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Consultando RAG (CV + GitHub)..."):
            try:
                respuesta = responder_usuario(prompt_input)
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                msg_error = "⌛ La API se encuentra ocupada temporalmente por alta demanda. Por favor reintenta tu pregunta en unos momentos."
                st.warning(msg_error)