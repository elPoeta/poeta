# P.O.E.T.A.

## Plan Maestro de Arquitectura, Especificaciones e Implementación

**Versión:** 1.1\
**Fecha:** 6 de noviembre de 2026\
**Plataforma objetivo:** Fedora Linux 44 · KDE Plasma · Wayland\
**Tipo de proyecto:** Agente personal de inteligencia artificial multimodal, local-first y extensible.

---

## Registro de cambios v1.0 → v1.1

- **Wake word:** el asistente se llama POETA. Se sustituye "JARVIS" por "POETA" como palabra de activación. Los modelos preentrenados de openWakeWord (por ejemplo "hey jarvis") ya no aplican: hay que entrenar un modelo personalizado (§8.2).
- **TTS:** Kokoro-82M en CPU (ONNX) pasa a ser la primera implementación. Piper con voces es_MX sube a candidato de la primera prueba A/B. Qwen3-TTS 0.6B queda como tercera opción (§8.3).
- **Interfaz `TTSEngine`:** motores intercambiables desde `voice.yaml` (§8.3).
- **Perfiles de ejecución:** se unifican conversación y respuesta hablada; Whisper y TTS residen en CPU y no usan VRAM (§14.1).
- **Pipeline de voz en streaming** con cancelación de eco e interrupciones (§15).
- **Fase 4 reordenada**, con benchmark A/B de voces y criterios de aceptación medibles (§17).
- **STT:** se medirán large-v3, large-v3-turbo y distil-large-v3 antes de fijar el modelo (§8.1).
- **Nuevo script:** `scripts/tts_benchmark.py` (§8.5).

---

# 1\. Visión general

P.O.E.T.A. (Personal Offline Executive & Thinking Assistant) será un agente personal de inteligencia artificial inspirado funcionalmente en JARVIS y HAL-9000. Su nombre propio, y la palabra con la que se le invoca, es **POETA**.

El objetivo es construir un asistente que pueda conversar naturalmente, escuchar, hablar, observar su entorno, interpretar gestos, navegar por Internet, utilizar aplicaciones, administrar información y ejecutar tareas automatizadas.

No será simplemente una interfaz conectada a un modelo de lenguaje. Será un sistema modular con un núcleo de razonamiento, memoria persistente, administración de modelos, herramientas externas, percepción multimodal y un sistema de permisos.

La prioridad será:

1. Funcionamiento local por defecto.
2. Privacidad y control de los datos.
3. Arquitectura independiente de los modelos.
4. Interacción natural mediante voz.
5. Percepción visual y reconocimiento de gestos.
6. Automatización supervisada.
7. Integración con servidores MCP.
8. Posibilidad de utilizar proveedores cloud cuando el usuario lo autorice.

## 1.1 Principios fundamentales

- Local-first: ningún proveedor cloud se utilizará automáticamente.
- Model-agnostic: el agente no dependerá de un modelo específico.
- Modularidad: cada subsistema tendrá una interfaz independiente.
- Interoperabilidad: uso de API OpenAI compatible y MCP.
- Seguridad: las acciones importantes requerirán autorización humana.
- Extensibilidad: posibilidad de incorporar nuevos modelos, herramientas y servicios.
- Persistencia: memoria y configuración independientes de los modelos.
- Observabilidad: registro de operaciones, errores, consumo y decisiones.

# 2\. Hardware y entorno de ejecución

## 2.1 Hardware disponible

| Componente | Especificación |
| --- | --- |
| Equipo | Lenovo ThinkPad P16 |
| CPU | Intel Core Ultra 7 255HX |
| Núcleos | 20 (8+12) |
| GPU | NVIDIA RTX PRO 2000 Blackwell Laptop |
| VRAM | 8 GB |
| RAM | 64 GB |
| Almacenamiento | NVMe de aproximadamente 1 TB |
| Espacio libre auditado | 658 GB |
| Sistema operativo | Fedora 44 |
| Kernel | Linux 7.2 |
| Escritorio | KDE Plasma |
| Sesión gráfica | Wayland |
| Audio | PipeWire 1.6.11 |

## 2.2 Entorno CUDA

Instalación existente:

- NVIDIA Driver 615.78.08.
- CUDA Toolkit 13.3.
- GCC 15.
- llama.cpp compilado con soporte CUDA.
- Arquitectura CUDA 120.

llama.cpp existente:

`~/AI/llama/llama.cpp`

Ejecutables disponibles:

- `llama-server`
- `llama-cli`
- `llama-gguf`

Versión auditada:

`0.7.0-dev`, commit `e486a4b55`.

**Decisión:** reutilizar la compilación existente. No recompilar ni sustituir componentes sin una necesidad técnica comprobada.

Las futuras actualizaciones de llama.cpp se realizarán de forma controlada, preservando el repositorio anterior y verificando los resultados de compilación.

# 3\. Organización del almacenamiento

Se conservará el entorno de IA existente.

Variables actuales:

bash

```bash
export AI_HOME="$HOME/AI"

export HF_HOME="$AI_HOME/models/huggingface"
export TORCH_HOME="$AI_HOME/models/torch"

export AI_MODELS="$AI_HOME/models"
export AI_VOICES="$AI_HOME/voices"
export AI_OUTPUT="$AI_HOME/output"
export AI_CACHE="$AI_HOME/cache"
```

P.O.E.T.A. tendrá un directorio independiente:

text

```text
~/AI/poeta/
```

No se duplicarán modelos que ya existan en el almacenamiento central.

## 3.1 Estructura de directorios propuesta

text

```text
~/AI/
│
├── models/
│   ├── huggingface/
│   ├── llama/
│   ├── whisper/
│   ├── fishspeech/              # Existente, no utilizado
│   └── ...
│
├── voices/
│   ├── reference/
│   ├── cloned/
│   └── poeta/
│
├── output/
│
└── poeta/
    │
    ├── .venv/
    ├── config/
    │   ├── settings.yaml
    │   ├── providers.yaml
    │   ├── models.yaml
    │   ├── voice.yaml
    │   ├── vision.yaml
    │   ├── audio.yaml
    │   ├── mcp.yaml
    │   └── permissions.yaml
    │
    ├── core/
    ├── providers/
    ├── models/
    ├── resources/
    ├── voice/
    ├── vision/
    ├── gestures/
    ├── memory/
    ├── mcp/
    ├── automation/
    ├── interface/
    ├── security/
    ├── monitoring/
    ├── tests/
    ├── scripts/
    ├── workspace/
    ├── logs/
    ├── main.py
    ├── pyproject.toml
    └── README.md
```

# 4\. Arquitectura general

P.O.E.T.A. se construirá mediante subsistemas independientes coordinados por un núcleo central.

text

```text
                         P.O.E.T.A.
                             |
                    +------------------+
                    |     AGENT CORE   |
                    +------------------+
                             |
          +------------------+------------------+
          |                  |                  |
     MODEL ROUTER        MEMORY             PERMISSIONS
          |                  |                  |
    +-----+------+       SQLite              Security
    |            |       Vector Store        Audit Log
   LOCAL        CLOUD
    |            |
 llama.cpp    NVIDIA NIM
              OpenAI API
              Otros
          |
     TOOL SYSTEM
          |
    MCP MANAGER
          |
    +-----+----------------------+
    |             |              |
 DuckDuckGo   Playwright     Filesystem
    |             |              |
  Search       Browser       Documents

          VOICE SYSTEM
               |
       +-------+-------+
       |               |
      STT             TTS
       |               |
   Whisper          Kokoro
                    Qwen TTS
                    Piper

          PERCEPTION
               |
       +-------+-------+
       |               |
     VISION          GESTURES
       |               |
    MediaPipe       MediaPipe
       |
     Webcam

          INTERFACE
               |
       KDE / HUD / Panel
```

# 5\. Núcleo central: Agent Core

Será el componente que coordine todas las operaciones.

Responsabilidades:

- Recibir mensajes del usuario.
- Mantener el contexto conversacional.
- Consultar el proveedor seleccionado.
- Administrar memoria.
- Solicitar herramientas.
- Validar permisos.
- Coordinar voz y visión.
- Gestionar tareas y operaciones asíncronas.
- Registrar eventos.
- Administrar interrupciones.

El núcleo no implementará directamente los proveedores ni las herramientas. Utilizará interfaces abstractas.

## 5.1 Model Router

Será responsable de seleccionar el modelo y proveedor configurado.

Debe soportar:

- Modelos locales.
- API OpenAI compatible.
- Streaming.
- Tool calling.
- Configuración de contexto.
- Parámetros de generación.
- Identificadores de modelos.
- Conexiones y errores.
- Gestión de disponibilidad.

La selección será manual desde el panel.

No habrá sustitución automática de un proveedor local por uno cloud.

# 6\. Modelos de inteligencia artificial

## 6.1 Modelos locales existentes

Se han identificado los siguientes modelos GGUF:

| Modelo | Tamaño aproximado |
| --- | --- |
| Qwen3.6-35B-A3B | 22,85 GB |
| GLM-4.7-Flash | 19,20 GB |
| Gemma 4 26B | 14,44 GB |
| GPT-OSS-20B | 13,79 GB |
| LFM2.5-8B-A1B | 6,95 GB |
| Nemotron 3.5 Bolt E4B | 6,18 GB |
| Qwen2.5-Coder-7B | 4,68 GB |

Existe además un `mmproj-BF16.gguf` para Qwen.

Estos tamaños representan los archivos almacenados, no el consumo total de inferencia.

## 6.2 Modelo principal

Candidato inicial:

**NVIDIA Nemotron 3.5 Bolt E4B Q4\_K\_M**

Ubicación:

text

```text
~/AI/models/llama/nvidia/
```

Debe probarse como modelo conversacional principal.

Se evaluarán:

- Calidad en español latinoamericano.
- Seguimiento de instrucciones.
- Tool calling.
- Latencia.
- Consumo de RAM.
- Consumo de VRAM.
- Rendimiento con llama.cpp.

Otros candidatos:

- Qwen3.6.
- GLM-4.7-Flash.
- Gemma 4.
- GPT-OSS-20B.
- LFM2.5.

No se establecerá un ganador definitivo hasta realizar pruebas.

## 6.3 Integración multimodal

Modelo candidato:

**LiquidAI LFM3-Omni-2B**

Todavía no está probado.

Se evaluará para:

- Comprensión de imágenes.
- Comprensión de audio.
- Percepción audiovisual.
- Interacción multimodal.
- Posible integración con visión en tiempo real.

No debe asumirse que todas sus capacidades estarán disponibles mediante llama.cpp. Se verificará el runtime y los requisitos específicos antes de integrarlo.

# 7\. Proveedores cloud

P.O.E.T.A. debe poder utilizar proveedores compatibles con la API de OpenAI.

Proveedores previstos:

| Proveedor | Endpoint |
| --- | --- |
| llama.cpp | http://127.0.0.1:8080/v1 |
| NVIDIA NIM | https://integrate.api.nvidia.com/v1 |
| OpenAI | https://api.openai.com/v1 |
| Otros | Configurable |

La implementación utilizará una interfaz común.

Ejemplo conceptual:

yaml

```yaml
defaults:
  provider: local
  cloud_fallback: false

providers:

  local:
    type: openai_compatible
    base_url: http://127.0.0.1:8080/v1
    api_key: local
    enabled: true

  nvidia:
    type: openai_compatible
    base_url: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY
    enabled: true

  openai:
    type: openai_compatible
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    enabled: false
```

Los modelos se seleccionarán manualmente desde el panel.

Las credenciales se almacenarán fuera de los archivos de configuración, preferentemente mediante el gestor de secretos del sistema.

## 7.1 Política de privacidad cloud

Toda operación que implique enviar información a un proveedor externo requerirá autorización.

La confirmación deberá mostrar:

- Proveedor.
- Modelo.
- Tipo de operación.
- Texto que se enviará.
- Contexto incluido.
- Documentos adjuntos.
- Resultados de herramientas incluidos.

No se permitirá enviar automáticamente:

- Memoria privada.
- Archivos locales.
- Imágenes de webcam.
- Audio del micrófono.
- Historial completo.

La autorización será específica para la operación.

# 8\. Sistema de voz

El sistema de voz será independiente del modelo conversacional.

Se dividirá en STT, Wake Word, TTS y Audio Manager.

## 8.1 Reconocimiento de voz (STT)

Tecnología seleccionada:

**faster-whisper**

Configuración conocida:

- Modelo Large-v3.
- Ejecución CPU.
- Español.
- Inferencia local.

Modelos encontrados:

- faster-whisper-large-v3.
- faster-distil-whisper-large-v3.
- faster-whisper-large-v3-turbo.

Se conservará la configuración que ya funciona.

Antes de fijar el modelo definitivo, en la Fase 4 se medirán con las mismas grabaciones en español:

| Modelo | Qué se mide |
| --- | --- |
| large-v3 | Referencia de calidad |
| large-v3-turbo | Latencia y calidad frente a la referencia |
| distil-large-v3 | Latencia y calidad (verificar su soporte de español antes de usarlo) |

Métricas: tiempo de transcripción por frase, tasa de error de palabras sobre un conjunto propio de frases y consumo de CPU. Whisper en CPU puede tardar varios segundos por frase con large-v3, lo que afecta directamente la fluidez de la conversación.

Configuración inicial propuesta (provisional hasta completar la medición):

yaml

```yaml
stt:
  engine: faster-whisper
  model: large-v3
  device: cpu
  language: es
  compute_type: int8
```

El tipo de cuantización definitivo respetará la configuración previamente probada.

## 8.2 Activación por voz

Palabra de activación:

**POETA**

Tecnología candidata:

openWakeWord.

Funciones:

- Detección local.
- Funcionamiento continuo.
- Baja latencia.
- Activación de la captura de voz.
- Reducción de falsas activaciones.

"POETA" no tiene modelo preentrenado: se necesitará entrenar un modelo personalizado de openWakeWord. Su flujo de entrenamiento se basa en generar muestras sintéticas con un TTS, por lo que debe verificarse en ese momento qué pipeline y qué voces soporta la versión vigente, incluyendo español.

Consideraciones propias de la palabra:

- Es de tres sílabas, con acentuación llana (po-E-ta), y es una palabra común del español: habrá riesgo de falsas activaciones en conversaciones o medios que la mencionen.
- Se mitiga con umbral de detección ajustable, ventana de confirmación y un conjunto de muestras negativas (palabras parecidas como "poesía", "poeta" dicho en frases normales, "planeta").
- Se necesitarán grabaciones reales de la voz del usuario, además de las sintéticas, para validar la tasa de falsos positivos y falsos negativos.

Alternativas si openWakeWord no alcanza la precisión requerida (a evaluar, no decididas):

- Variante con prefijo ("oye Poeta") para ganar robustez.
- Otro motor de wake word con palabras personalizadas, revisando licencia y ejecución 100 % local.
- Detección por VAD + transcripción corta con Whisper, con mayor coste de CPU.

Criterio de aceptación: medir falsas activaciones por hora en un entorno normal y porcentaje de activaciones correctas; los umbrales concretos se fijarán con las primeras mediciones.

## 8.3 Síntesis de voz (TTS)

Objetivo:

Voz masculina natural en español latinoamericano.

Fish Speech queda expresamente descartado.

Modelos disponibles:

**Qwen3-TTS**

Se encontraron:

- Qwen3-TTS Tokenizer 12Hz.
- Qwen3-TTS 1.7B Base.
- Qwen3-TTS 1.7B CustomVoice.

Tamaño aproximado total: 9,25 GB.

Se conservarán para experimentación, pero no serán la primera opción obligatoria.

### Candidatos

| Modelo | Parámetros | Ejecución | Prioridad |
| --- | --- | --- | --- |
| Kokoro-82M | 82M | CPU (ONNX, `kokoro-onnx`) | Primera implementación y primera prueba A/B |
| Piper (voces es_MX) | Variable según voz | CPU | Segundo candidato de la primera prueba A/B |
| Qwen3-TTS 0.6B | 600M | A definir tras medir | Solo si ninguno de los anteriores convence |
| Qwen3-TTS 1.7B | 1,7B | A definir tras medir | Alternativa de calidad, conservado para pruebas |

**Kokoro-82M:** es la opción más conveniente en recursos. Con 82M parámetros corre por encima del tiempo real en CPU, deja los 8 GB de VRAM íntegros para el LLM y puede permanecer residente en RAM (orden de 0,5 GB, a medir). Requiere `espeak-ng` instalado desde dnf para la fonemización. Riesgo principal: las voces en español son pocas, con pocas opciones masculinas, y no están orientadas específicamente al español latinoamericano. Los nombres de voces y su disponibilidad deben verificarse en la versión vigente.

**Piper:** tiene voces masculinas en español de México y consumo todavía menor. Por requisito de acento, entra como candidato de primera prueba y no solo como respaldo.

**Qwen3-TTS 0.6B:** se evalúa únicamente para comparar expresividad si Kokoro y Piper no resultan satisfactorios.

### Interfaz TTSEngine

Todos los motores se implementarán como adaptadores de una interfaz común, seleccionables desde `voice.yaml`:

text

```text
TTSEngine
├── warmup()                          # Carga el modelo y sintetiza una frase corta
├── synthesize_stream(text) -> Iterator[AudioChunk]
├── stop()                            # Cancela la síntesis en curso
├── sample_rate (propiedad)
└── close()                           # Libera recursos
```

Adaptadores iniciales: `KokoroEngine`, `PiperEngine`, `QwenTTSEngine`.

Ejemplo conceptual de `voice.yaml`:

yaml

```yaml
tts:
  engine: kokoro          # kokoro | piper | qwen3
  resident: true          # Mantener cargado en RAM
  kokoro:
    runtime: onnx
    device: cpu
    voice: TBD            # Se fija tras la prueba A/B
    speed: 1.0
  piper:
    voice: TBD            # Por ejemplo, una voz es_MX masculina
  qwen3:
    model: 0.6B
    device: TBD
wake_word:
  phrase: poeta
  engine: openwakeword
  model_path: ~/AI/poeta/models/wakeword/poeta.onnx
  threshold: TBD          # Se calibra con mediciones
```

### Requisitos del sintetizador

- Voz masculina.
- Español latinoamericano.
- Pronunciación natural.
- Baja latencia.
- Audio interrumpible.
- Ejecución local.
- Independencia del LLM.

La elección definitiva se realizará mediante pruebas auditivas comparables.

## 8.4 Audio Manager

Será responsable de:

- Captura del micrófono.
- Reproducción.
- Control de volumen.
- Gestión de dispositivos.
- Cancelación de eco.
- Detección de actividad de voz.
- Interrupciones.
- Enrutamiento de audio.

Se integrará con PipeWire.

## 8.5 Prueba A/B de voces

La elección del TTS se decidirá por oído y por latencia, con el script `scripts/tts_benchmark.py`.

Procedimiento:

1. Preparar 15-20 frases de uso real: respuestas cortas, números, fechas, horas, nombres propios, anglicismos técnicos (por ejemplo "llama.cpp", "Wayland", "MCP") y frases largas.
2. Sintetizar las mismas frases con cada motor y voz candidata.
3. Medir por frase: latencia al primer audio, tiempo total de síntesis, factor de tiempo real (RTF) y memoria utilizada.
4. Guardar los WAV en `~/AI/output/tts_benchmark/` con nombres que identifiquen motor y voz.
5. Escuchar a ciegas y puntuar naturalidad, acento y pronunciación.

Criterio de decisión: el motor que cumpla el requisito de acento y naturalidad con la menor latencia al primer audio.

## 8.6 Cancelación de eco e interrupciones

Para que el usuario pueda interrumpir a POETA mientras habla se adoptará una de estas estrategias, a probar en la Fase 4:

- Cancelación de eco con el módulo `module-echo-cancel` de PipeWire, de modo que el micrófono siga escuchando durante la reproducción.
- Alternativa más simple: silenciar la captura general mientras se reproduce y mantener activa solo la detección de la palabra POETA.

Al detectarse una interrupción, `stop()` del TTS y la cola de audio se vacían de inmediato.

# 9\. Sistema de visión y gestos

Se utilizará una única webcam para ambas funciones.

Dispositivo identificado:

text

```text
/dev/video0
```

Formatos disponibles:

- MJPEG hasta 2560x1440.
- YUYV 640x480.

Dispositivo adicional:

text

```text
/dev/video2
```

Ofrece vídeo monocromático de 640x360 y podría corresponder al sensor infrarrojo.

La identificación funcional definitiva debe realizarse antes de implementarlo.

## 9.1 Arquitectura de captura compartida

Una única instancia de captura distribuirá los fotogramas a los subsistemas.

text

```text
                 WEBCAM
                    |
              CAMERA MANAGER
                    |
             FRAME DISTRIBUTOR
                    |
          +---------+---------+
          |                   |
       VISION              GESTURES
          |                   |
      MediaPipe           MediaPipe
          |                   |
    Face Detection       Hand Tracking
    Object Detection     Gesture Recognition
    Scene Analysis       Motion Analysis
```

No se abrirá la webcam independientemente desde cada módulo.

El sistema deberá controlar:

- Frecuencia de captura.
- Resolución.
- Distribución de fotogramas.
- Sincronización.
- Uso de CPU.
- Liberación de recursos.

## 9.2 MediaPipe

Tecnología principal:

MediaPipe Tasks.

Funciones previstas:

- Detección de manos.
- Seguimiento de landmarks.
- Reconocimiento de gestos.
- Detección facial.
- Seguimiento de presencia.
- Reconocimiento de postura.

Se desarrollará un sistema propio inspirado en Barehands.

Repositorio de referencia:

[https://github.com/jaredrhod/barehands](<https://github.com/jaredrhod/barehands>)

No se dependerá directamente de ese proyecto.

## 9.3 Gestos

Gestos iniciales:

| Gesto | Acción propuesta |
| --- | --- |
| Palma abierta | Pausar o detener |
| Pulgar arriba | Confirmar |
| Pulgar abajo | Cancelar |
| Índice señalando | Seleccionar |
| Mano abierta mantenida | Activar interacción |
| Movimiento horizontal | Navegación |

Los gestos deberán tener calibración, umbrales temporales y protección contra activaciones accidentales.

Las acciones sensibles no se autorizarán exclusivamente mediante gestos.

# 10\. Sistema MCP

MCP será una capacidad fundamental del agente.

P.O.E.T.A. implementará un cliente MCP propio utilizando el SDK oficial de Python.

Se soportarán:

- stdio.
- Streamable HTTP.

El cliente deberá gestionar conexiones, sesiones, herramientas, errores y permisos.

## 10.1 Servidores iniciales

| Servidor | Objetivo |
| --- | --- |
| DuckDuckGo Search | Búsquedas web |
| Playwright | Navegación web |
| Filesystem | Gestión de archivos |
| Fetch | Obtención de contenido web |
| SQLite | Consultas de bases de datos |
| GitHub | Repositorios y código |

Los servidores deberán verificarse individualmente antes de instalarlos.

DuckDuckGo se integrará mediante un servidor MCP compatible de terceros.

Playwright MCP permitirá:

- Navegar.
- Leer páginas.
- Interactuar con elementos.
- Extraer información.
- Gestionar sesiones web.

## 10.2 MCP Manager

Módulos previstos:

text

```text
mcp/
├── client_manager.py
├── server_registry.py
├── tool_discovery.py
├── tool_adapter.py
├── tool_permissions.py
├── connection_manager.py
└── audit_log.py
```

Funciones:

- Registrar servidores.
- Conectar y desconectar.
- Descubrir herramientas.
- Consultar esquemas.
- Validar parámetros.
- Ejecutar herramientas.
- Registrar resultados.
- Gestionar errores.
- Habilitar y deshabilitar herramientas.

## 10.3 Seguridad MCP

Los servidores MCP no tendrán permisos ilimitados.

Se establecerá una lista de servidores autorizados.

Las herramientas se clasificarán por riesgo:

| Nivel | Operación | Permiso |
| --- | --- | --- |
| 1 | Consultas de lectura | Automático |
| 2 | Navegación y extracción | Automático |
| 3 | Modificación de archivos | Confirmación |
| 4 | Acciones sensibles o irreversibles | Confirmación explícita |

Los resultados de herramientas y páginas web serán considerados datos no confiables.

Nunca podrán modificar las instrucciones fundamentales del agente.

# 11\. Sistema de memoria

La memoria será independiente de los modelos.

Debe sobrevivir a:

- Cambios de proveedor.
- Cambios de modelo.
- Reinicios.
- Actualizaciones.
- Cierres de sesión.

## 11.1 Tipos de memoria

### Memoria conversacional

Registro de conversaciones y mensajes.

### Memoria episódica

Registro de acontecimientos y tareas realizadas.

### Memoria semántica

Conocimientos y relaciones persistentes.

### Memoria de preferencias

Configuraciones y preferencias expresamente autorizadas.

### Memoria operativa

Historial de herramientas, acciones y resultados.

## 11.2 Tecnologías

Primera etapa:

- SQLite.
- FTS5 para búsqueda textual.
- Sistema de embeddings desacoplado.

Más adelante:

- Base vectorial local.
- Recuperación semántica.
- Búsqueda híbrida.
- Consolidación de recuerdos.
- Expiración configurable.

No se enviará memoria privada a proveedores cloud sin autorización.

# 12\. Automatización del sistema

P.O.E.T.A. deberá poder interactuar con Fedora y KDE.

Capacidades previstas:

- Abrir aplicaciones.
- Gestionar ventanas.
- Ejecutar comandos autorizados.
- Administrar archivos.
- Controlar servicios.
- Navegar por Internet.
- Gestionar procesos.
- Automatizar tareas repetitivas.

## 12.1 Wayland y KDE

La integración deberá respetar las particularidades de Wayland.

Se evaluarán:

- XDG Desktop Portal.
- KDE scripting.
- D-Bus.
- KWin.
- PipeWire.
- Atajos de teclado autorizados.

No se asumirá que las técnicas de automatización de X11 funcionan igual en Wayland.

## 12.2 Sistema de permisos

Las acciones se clasificarán según impacto.

Ejemplos:

| Acción | Política |
| --- | --- |
| Abrir una aplicación | Automática |
| Consultar un archivo autorizado | Automática |
| Crear un documento | Según carpeta y política |
| Modificar archivos existentes | Confirmación |
| Ejecutar comandos | Lista permitida y confirmación |
| Instalar paquetes | Confirmación |
| Eliminar información | Confirmación explícita |
| Operaciones privilegiadas | Nunca automáticas |

La autorización será validada por el núcleo y no por el modelo de lenguaje.

# 13\. Interfaz gráfica

Se desarrollará un panel propio de P.O.E.T.A.

Objetivos:

- Estado del agente.
- Modelo activo.
- Proveedor activo.
- Consumo de GPU.
- Consumo de RAM.
- Estado de audio.
- Estado de webcam.
- Gestión MCP.
- Historial de conversaciones.
- Memoria.
- Permisos.
- Automatizaciones.

## 13.1 HUD

Se evaluará una interfaz visual de estilo futurista, inspirada en las interfaces de ciencia ficción (JARVIS entre ellas).

Características:

- Indicadores de actividad.
- Visualización de audio.
- Estado de percepción.
- Información contextual.
- Animaciones.
- Indicador de escucha.
- Indicador de procesamiento.
- Indicador de respuesta.

Tecnologías candidatas:

- Python para el backend.
- Qt/PySide6 para la aplicación de escritorio.
- QML para interfaces nativas.
- Three.js para visualizaciones 3D si se utiliza una interfaz web.

La interfaz no deberá bloquear el agente cuando se cierre.

# 14\. Administración de recursos

Los 8 GB de VRAM son el principal límite de diseño.

P.O.E.T.A. tendrá un Resource Manager.

Responsabilidades:

- Consultar memoria disponible.
- Supervisar modelos cargados.
- Controlar carga y descarga.
- Evitar conflictos.
- Administrar prioridades.
- Supervisar procesos.
- Registrar latencia.
- Evitar cargas simultáneas incompatibles.

## 14.1 Perfiles de ejecución

### Perfil conversación (incluye respuesta hablada)

- LLM principal (Nemotron candidato) en GPU.
- faster-whisper residente en CPU.
- TTS (Kokoro o Piper) residente en CPU.
- Wake word residente en CPU.
- Uso de VRAM por voz: cero.

Se elimina el ciclo de carga y descarga del TTS: al ser ligero y correr en CPU, permanece cargado, lo que reduce la latencia de la primera frase y la complejidad del Resource Manager. Whisper y el TTS no compiten entre sí, porque POETA primero escucha y después habla; esto se verificará midiendo el uso de CPU durante la interrupción.

Si se elige Qwen3-TTS como motor, este perfil deberá revisarse tras medir su consumo.

### Perfil percepción

- LFM3 activo cuando se necesite.
- Captura de cámara.
- Procesamiento multimodal.
- LLM principal suspendido o descargado si es necesario.

### Perfil cloud

- Modelo cloud seleccionado manualmente.
- Confirmación de transferencia.
- Inferencia remota.
- Recursos locales liberados si conviene.

No se establecerán límites de VRAM definitivos hasta medir cada modelo en ejecución.

# 15\. Wake Word y experiencia de conversación

La experiencia objetivo será:

1. El usuario dice POETA.
2. P.O.E.T.A. detecta la activación.
3. Se inicia la captura de voz.
4. Whisper transcribe.
5. El agente interpreta la petición.
6. Consulta memoria y herramientas cuando corresponde.
7. Solicita autorización si es necesario.
8. Genera respuesta.
9. El TTS produce audio.
10. El usuario puede interrumpirlo hablando.

## 15.1 Pipeline de respuesta hablada en streaming

POETA empieza a hablar con la primera oración, sin esperar la respuesta completa del LLM:

text

```text
LLM (streaming)
      |
Divisor por oraciones
      |
Cola de síntesis  -->  TTSEngine.synthesize_stream()
      |
Cola de audio     -->  PipeWire
```

Reglas:

- El divisor corta por signos de puntuación y evita cortar abreviaturas, números y decimales.
- Texto no hablable (bloques de código, URLs, tablas) se filtra o se resume antes de entrar a la cola de síntesis.
- Ante una interrupción se vacían ambas colas y se cancela la generación del LLM si corresponde.
- Los eventos de estado (escuchando, procesando, hablando) se publican para el panel y el HUD.

El sistema deberá gestionar:

- Detección de fin de frase.
- Silencios.
- Interrupciones.
- Errores de reconocimiento.
- Cancelación.
- Reanudación.
- Respuestas parciales.

# 16\. Tecnologías propuestas

| Subsistema | Tecnología |
| --- | --- |
| Lenguaje principal | Python |
| Entorno | uv |
| Runtime LLM | llama.cpp |
| API cloud | OpenAI-compatible |
| STT | faster-whisper |
| Wake Word | openWakeWord |
| TTS | Kokoro / Qwen3-TTS / Piper |
| Visión | OpenCV |
| Gestos | MediaPipe |
| MCP | MCP Python SDK |
| Navegación | Playwright |
| Memoria | SQLite |
| Búsqueda textual | SQLite FTS5 |
| Interfaz | PySide6 / QML |
| Sistema operativo | Fedora 44 |
| Audio | PipeWire |
| Comunicación | asyncio |
| Configuración | YAML |
| Pruebas | pytest |

## 16.1 Python

Existe Python 3.12.13 instalado mediante uv.

También están disponibles Python 3.13 y 3.14.

Se utilizará inicialmente Python 3.12 para el entorno independiente, por su compatibilidad con el ecosistema de IA.

Directorio:

text

```text
~/AI/poeta/.venv/
```

No se instalarán dependencias en el Python global.

Las versiones se fijarán y probarán progresivamente.

# 17\. Fases de implementación

La construcción se realizará por etapas verificables.

## Fase 0 — Preparación

Objetivos:

- Crear el proyecto.
- Crear entorno Python.
- Crear estructura.
- Configurar variables.
- Crear archivos YAML.
- Preparar registros.
- Verificar dependencias.

Resultado:

Proyecto P.O.E.T.A. inicializado sin modificar otros proyectos.

## Fase 1 — Core y Model Router

Objetivos:

- Implementar Agent Core.
- Implementar proveedor OpenAI compatible.
- Integrar llama.cpp.
- Integrar NVIDIA NIM.
- Añadir selección manual.
- Implementar streaming.
- Añadir registro de errores.

Resultado:

Conversación funcional mediante un modelo local y uno cloud.

Criterio de aceptación:

Cambiar de proveedor desde configuración o interfaz sin modificar el código del agente.

## Fase 2 — Permisos y seguridad

Objetivos:

- Implementar Permission Manager.
- Crear clasificación de riesgos.
- Implementar confirmaciones.
- Crear registro de auditoría.
- Controlar transferencias cloud.

Resultado:

Ninguna operación sensible podrá ejecutarse sin autorización.

Criterio de aceptación:

Comprobar que una petición cloud se bloquea hasta recibir autorización.

## Fase 3 — Memoria

Objetivos:

- SQLite.
- Historial.
- Memoria episódica.
- Memoria de preferencias.
- Recuperación contextual.
- Búsqueda.

Resultado:

P.O.E.T.A. conservará contexto entre sesiones y cambios de modelo.

## Fase 4 — Voz

Orden de trabajo:

1. **Audio Manager:** captura, reproducción, VAD y gestión de dispositivos con PipeWire.
2. **faster-whisper:** integración y medición de latencia y error con large-v3, large-v3-turbo y distil-large-v3. Fijar el modelo según los resultados.
3. **TTS Manager:** interfaz `TTSEngine`, adaptadores Kokoro (ONNX, CPU) y Piper (voces es_MX), y prueba A/B con `scripts/tts_benchmark.py`. Qwen3-TTS 0.6B solo si ninguno convence.
4. **Pipeline en streaming:** LLM → divisor por oraciones → cola de síntesis → cola de audio, con interrupciones y cancelación de eco (§8.6, §15.1).
5. **Wake word POETA:** recolección de muestras, entrenamiento del modelo personalizado de openWakeWord y calibración del umbral. Va al final por ser la parte más incierta.

Resultado:

Conversación hablada completamente local.

Criterios de aceptación (valores objetivo iniciales, a ajustar con las mediciones reales):

- Primer audio en menos de aproximadamente 1,5 s tras completarse la primera oración del LLM.
- Interrupción efectiva en menos de aproximadamente 300 ms.
- Cero VRAM usada por voz.
- TTS y Whisper residentes sin degradar la latencia del LLM.
- Falsas activaciones de la wake word dentro del límite que se fije tras las primeras mediciones.

## Fase 5 — Visión y gestos

Objetivos:

- Identificar dispositivos.
- Implementar Camera Manager.
- Implementar captura compartida.
- Integrar MediaPipe.
- Crear Hand Tracking.
- Crear Gesture Engine.
- Implementar calibración.

Resultado:

P.O.E.T.A. podrá observar la webcam y reconocer gestos sin abrir múltiples instancias de captura.

## Fase 6 — MCP

Objetivos:

- Implementar cliente MCP.
- Implementar registro de servidores.
- Añadir descubrimiento.
- Añadir validación de herramientas.
- Integrar DuckDuckGo.
- Integrar Playwright.
- Integrar Filesystem.
- Implementar permisos y auditoría.

Resultado:

P.O.E.T.A. podrá utilizar herramientas externas de forma controlada.

## Fase 7 — Automatización KDE

Objetivos:

- Integrar D-Bus.
- Integrar XDG Desktop Portal.
- Gestionar aplicaciones.
- Gestionar ventanas.
- Ejecutar acciones autorizadas.
- Implementar automatizaciones.

Resultado:

Interacción supervisada con Fedora y KDE.

## Fase 8 — Interfaz y HUD

Objetivos:

- Panel principal.
- Selector de modelos.
- Administrador MCP.
- Monitor de recursos.
- Panel de voz.
- Panel de visión.
- Historial.
- Gestión de permisos.

Resultado:

Una interfaz unificada para controlar todo P.O.E.T.A.

## Fase 9 — Integración y optimización

Objetivos:

- Pruebas de carga.
- Pruebas de estabilidad.
- Pruebas de memoria.
- Medición de latencia.
- Optimización de GPU.
- Pruebas de interrupciones.
- Pruebas de seguridad.
- Inicio automático supervisado.

Resultado:

Sistema integrado y preparado para uso cotidiano.

# 18\. Recursos necesarios

## 18.1 Software

Ya disponible:

- Fedora 44.
- CUDA 13.3.
- GCC 15.
- llama.cpp.
- Docker.
- Podman.
- FFmpeg.
- Git.
- CMake.
- Ninja.
- uv.
- Modelos GGUF.
- Modelos Whisper.
- Qwen3-TTS.
- Fish Speech (archivos existentes, descartado para el proyecto).

Por evaluar o instalar:

- PySide6.
- OpenCV.
- MediaPipe.
- faster-whisper en el entorno P.O.E.T.A.
- openWakeWord (con entrenamiento de modelo personalizado "poeta").
- Kokoro (`kokoro-onnx`) y `espeak-ng` desde dnf.
- Piper y una voz es_MX masculina.
- MCP Python SDK.
- Playwright.
- Dependencias específicas de Qwen3-TTS si se utiliza.

## 18.2 Hardware

No se requiere hardware adicional para comenzar.

La webcam integrada y el micrófono del portátil serán los dispositivos iniciales.

## 18.3 Almacenamiento

Se aprovecharán los modelos existentes.

No se descargarán modelos adicionales hasta determinar que son necesarios.

Se mantendrá una separación entre:

- Modelos.
- Código.
- Configuración.
- Caché.
- Memoria.
- Registros.
- Resultados.

# 19\. Seguridad y privacidad

La seguridad será una característica transversal.

Requisitos obligatorios:

1. Local por defecto.
2. Cloud nunca automático.
3. Confirmación de transferencias externas.
4. Secretos fuera del código.
5. Servidores MCP autorizados.
6. Herramientas con permisos individuales.
7. Restricción de acceso a archivos.
8. Validación de argumentos.
9. Registro de operaciones.
10. Protección contra instrucciones maliciosas procedentes de páginas web.
11. Confirmación de acciones irreversibles.
12. Separación entre razonamiento y ejecución.

Los modelos no podrán modificar sus propios permisos.

El agente no tendrá acceso ilimitado al sistema operativo.

# 20\. Pruebas y criterios de calidad

Cada módulo deberá disponer de pruebas independientes.

## Model Router

- Conexión local.
- Conexión cloud.
- Streaming.
- Selección manual.
- Manejo de errores.
- Ausencia de fallback automático.

## Voz

- Reconocimiento español.
- Latencia.
- Calidad de voz.
- Interrupciones.
- Consumo de recursos.

## Visión

- Detección estable.
- Seguimiento de manos.
- Reconocimiento de gestos.
- Uso compartido de webcam.

## MCP

- Conexiones.
- Descubrimiento.
- Ejecución.
- Validación.
- Permisos.
- Desconexión.
- Auditoría.

## Seguridad

- Bloqueo de operaciones no autorizadas.
- Bloqueo de transferencias cloud.
- Restricción de herramientas.
- Protección de secretos.
- Validación de acciones sensibles.

# 21\. Orden de prioridades

La implementación debe respetar este orden:

| Prioridad | Componente |
| --- | --- |
| P0 | Estructura y entorno |
| P0 | Agent Core |
| P0 | Model Router |
| P0 | Permisos y privacidad |
| P1 | Memoria |
| P1 | STT y TTS |
| P1 | Wake Word |
| P1 | MCP |
| P2 | Visión |
| P2 | Gestos |
| P2 | Automatización KDE |
| P3 | HUD |
| P3 | Optimización avanzada |

La interfaz visual no debe retrasar el funcionamiento del núcleo.

# 22\. Definición de terminado

P.O.E.T.A. se considerará una primera versión funcional cuando pueda:

- Mantener una conversación local.
- Seleccionar modelos manualmente.
- Utilizar NVIDIA mediante API OpenAI compatible.
- Solicitar autorización antes de transmitir datos.
- Recordar información entre sesiones.
- Activarse al decir POETA.
- Transcribir voz en español.
- Responder con voz masculina natural.
- Interrumpir su propia reproducción.
- Utilizar una webcam compartida.
- Reconocer gestos.
- Consultar Internet mediante MCP.
- Navegar con Playwright MCP.
- Ejecutar herramientas autorizadas.
- Automatizar operaciones supervisadas en KDE.
- Mostrar su estado desde un panel central.

# 23\. Decisiones arquitectónicas definitivas

| Área | Decisión |
| --- | --- |
| Nombre | P.O.E.T.A. |
| Sistema | Fedora 44 |
| Lenguaje | Python |
| Python inicial | 3.12 |
| Entorno | uv + venv |
| LLM local | Nemotron candidato inicial |
| Runtime local | llama.cpp |
| API cloud | OpenAI compatible |
| Proveedor cloud candidato | NVIDIA NIM |
| Selección de modelos | Manual |
| Proveedor predeterminado | Local |
| Autorización cloud | Cada operación |
| STT | faster-whisper en CPU (large-v3 provisional; medir turbo y distil) |
| Wake Word | POETA (openWakeWord, modelo personalizado) |
| TTS primera implementación | Kokoro-82M, ONNX, CPU, residente |
| TTS candidato de primera prueba A/B | Piper (voces es_MX) |
| TTS alternativo | Qwen3-TTS 0.6B (solo si los anteriores no convencen) |
| Qwen3-TTS 1.7B | Conservado para pruebas |
| Interfaz de TTS | `TTSEngine` con adaptadores intercambiables |
| Pipeline de voz | Streaming por oraciones con interrupción |
| Visión | MediaPipe |
| Webcam | Una sola instancia compartida |
| Gestos | Implementación propia |
| Memoria | SQLite |
| Herramientas | MCP |
| Búsqueda | DuckDuckGo MCP |
| Navegación | Playwright MCP |
| Interfaz | PySide6 / QML |
| Automatización | KDE / D-Bus / Portals |
| Seguridad | Supervisión humana |

# 24\. Primer hito de desarrollo

El primer objetivo concreto será crear:

text

```text
~/AI/poeta/
```

Con:

1. Entorno Python independiente.
2. Configuración centralizada.
3. Agent Core inicial.
4. Model Router.
5. Proveedor llama.cpp.
6. Proveedor OpenAI compatible.
7. Configuración NVIDIA.
8. Permission Manager.
9. Registro de eventos.
10. Pruebas automatizadas.

No se instalarán todos los componentes de una vez.

Cada fase tendrá sus propias dependencias, pruebas y criterios de aceptación.

**La regla principal del proyecto será construir, probar y validar cada subsistema antes de incorporar el siguiente.**

---

**FIN DEL DOCUMENTO — P.O.E.T.A. MASTER PLAN v1.1**

