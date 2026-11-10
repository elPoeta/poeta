# P.O.E.T.A.

Personal Offline Executive & Thinking Assistant. Plan completo en `docs/POETA_v1.1.md`.

Estado: **Fase 0** (estructura, configuracion, registro de eventos, diagnostico). El Agent Core y el Model Router llegan en la Fase 1.

## Puesta en marcha

```bash
cd ~/AI/poeta
bash scripts/setup.sh      # crea .venv (Python 3.12 con uv), instala, corre pruebas y diagnostico
source .venv/bin/activate
python scripts/doctor.py   # diagnostico del entorno (solo lectura)
python main.py             # carga y valida la configuracion
```

## Fase 1: chat con el modelo local

```bash
python scripts/run_llama_server.py --dry-run   # revisa el comando; quita --dry-run para lanzarlo
python scripts/run_llama_server.py --model CLAVE   # otro modelo de models.yaml (cada uno con su bloque `server`)
python main.py chat                            # en otra terminal
```

Comandos del chat: `/status`, `/provider NOMBRE [MODELO]`, `/model NOMBRE`, `/reset`, `/quit`.
Para NVIDIA: exporta `NVIDIA_API_KEY`, y usa `/provider nvidia MODELO`. Antes de enviar nada se muestra el contenido exacto y se pide confirmacion en cada operacion.
Nunca hay cambio automatico de proveedor: si el local falla, el error se muestra y se queda ahi.

## Fase 2: permisos y auditoria

Todo lo que tenga impacto pasa por `PermissionManager` (`poeta/security/`): transferencias cloud, archivos, comandos y herramientas. Falla cerrado: accion no catalogada, sin interfaz de confirmacion, error, tiempo agotado o fallo al auditar => se deniega.

- **Niveles** (`config/permissions.yaml`): 1-2 automaticos, 3 confirmacion, 4 confirmacion explicita (escribir `CONFIRMAR`), `never` para operaciones privilegiadas. El YAML puede endurecer la politica pero no relajarla por debajo de los minimos (`MIN_LEVEL`).
- **Cloud**: cada transferencia se confirma por separado y se envia exactamente lo que se mostro. Con imagenes o audio la confirmacion es explicita.
- **Archivos**: solo dentro de `file_access.allowed_dirs` (por defecto, `workspace/`). `config/`, `logs/` y `.env` estan protegidos incluso con consentimiento. Los enlaces simbolicos se resuelven antes de comprobar.
- **Comandos**: solo los de `commands.allowlist` (vacia por defecto), como lista argv, y con confirmacion.
- **Auditoria**: `logs/audit.jsonl`, con cadena de hashes. Registra metadatos y huellas, nunca el contenido ni secretos.

```bash
python main.py audit             # ultimos eventos
python main.py audit --verify    # comprueba la cadena de hashes
```

Si la cadena esta rota, `main.py chat` se niega a arrancar hasta que muevas el archivo a otro nombre (por ejemplo tras un corte de luz a mitad de una escritura).

Limites: la cadena de hashes detecta ediciones y borrados intermedios, no protege contra quien reescriba todo el archivo ni detecta que se trunquen las ultimas entradas. El gestor decide sobre lo que se le pide; que las herramientas lo consulten antes de actuar es responsabilidad de las fases 6 y 7.

## Fase 3: memoria

SQLite + FTS5 en `data/memory.db` (privada, 0600; el agente no puede leerla ni escribirla). Independiente del modelo: una conversacion sigue siendo una aunque cambies de modelo o de proveedor.

- **Se guarda solo:** cada intercambio de chat (`save_conversations`). Un turno que falla sin texto no se guarda; uno cortado a mitad se guarda como parcial.
- **Se guarda por orden tuya:** notas (`/remember`) y preferencias (`/prefer clave=valor`). El modelo no decide que recordar.
- **Se usa:** antes de cada turno se buscan notas, preferencias y fragmentos de conversaciones anteriores relacionados con lo que escribes, y se agregan al prompt del sistema como *datos, no instrucciones*. No se guardan en el historial.
- **Nunca a la nube por defecto** (`memory.context.inject_into_cloud: false`). La conversacion de un turno cloud se guarda igual, pero localmente.
- **Retencion** (`retention_days`) y expiracion por nota: se aplican al arrancar.

```bash
python main.py chat --continue      # retoma la ultima conversacion
/conversations   /resume ID         # elegir otra
/remember TEXTO  /memories [CONSULTA]  /forget ID
/prefer CLAVE=VALOR  /unprefer CLAVE
/incognito                          # esta sesion no se guarda (la memoria existente se sigue usando)
python main.py memory stats | export ARCHIVO.json | wipe
```

Borrado: `/forget`, `/unprefer`, la expiracion y `wipe` sobrescriben las paginas (`secure_delete`) y truncan el WAL, y se comprueba sobre los bytes crudos del archivo. En un SSD no hay garantia de borrado fisico: para eso, cifrado de disco (LUKS).

Limites actuales: la busqueda es por palabras (FTS5), no por significado: una nota se encuentra si comparte palabras con lo que escribes. La base **no esta cifrada** por si misma. La busqueda vectorial y la consolidacion de recuerdos quedan para una etapa posterior (`poeta/memory/embeddings.py` solo define la interfaz).

## Estructura

- `config/` — YAML (settings, providers, models, voice, vision, audio, mcp, permissions). `null` significa "por decidir tras medir".
- `poeta/` — paquete con un subpaquete por subsistema (`core`, `providers`, `voice`, ...). Vacios hasta su fase.
- `scripts/` — `setup.sh`, `doctor.py`, `tts_benchmark.py`.
- `tests/` — pruebas con pytest.
- `logs/`, `workspace/` — salida local, fuera de git.

## Invariantes de privacidad validados al cargar la configuracion

- `cloud_fallback` siempre `false`.
- El proveedor por defecto debe ser local.
- Los proveedores cloud no pueden llevar la clave en el YAML (solo `api_key_env`).
- `permissions.cloud.require_confirmation` siempre `true`.

Las claves cloud van en el gestor de secretos del sistema o, en su defecto, en `.env` (ignorado por git).

## Fase 4 (en curso): voz

Instalar extras de voz (faster-whisper, numpy, sounddevice, kokoro-onnx) y las librerías CUDA 12 que necesita CTranslate2:

    uv pip install -e '.[voice,cuda]'

- `poeta/voice/stt.py`: transcriptor local con la configuración probada en vozboard (beam 5, VAD 500 ms, español, pesos en `$AI_MODELS/whisper`).
- `python scripts/stt_benchmark.py --audio-dir ~/AI/output/vozboard/audio --combo large-v3:cuda:float16 --combo large-v3-turbo:cuda:int8_float16 --combo large-v3-turbo:cpu:int8`: compara modelos y dispositivos (latencia, WER si hay `.txt` de referencia junto al audio, VRAM). Medir con el servidor del LLM encendido.
- `python scripts/tts_benchmark.py kokoro|piper ...`: A/B de voces (requiere `kokoro-onnx` o `piper-tts`).

### Audio (Fase 4b)

Requisito del sistema: `sudo dnf install portaudio` (PipeWire lo atiende por su capa de compatibilidad).
Con el micrófono interno y parlantes, POETA ignora el micrófono mientras habla (`echo_cancel.strategy: mute_during_playback`).
Pruebas en vivo: `python scripts/mic_test.py devices|level|listen|echo`.

### Conversación por voz (Fase 4d)

    python main.py voice            # Enter = hablar; escribir texto también funciona; /quit sale
    python main.py voice --always   # escucha continuamente (opt-in, sin palabra de activación todavía)

Enter mientras POETA habla la interrumpe. Tras cada turno se muestran las latencias (STT, primer texto, primer audio).

### Permisos (configurables, con avisos)

`config/permissions.yaml` se puede relajar a tu gusto: si un nivel queda por debajo de lo recomendado, POETA arranca igual,
muestra `Aviso de permisos: ... Recomendado: N` y deja constancia en la auditoría (`permissions_relaxed`).
Siguen rechazándose los errores de forma y `privileged` (siempre `never`). En modo voz, las confirmaciones se responden
escribiendo en la misma terminal (la respuesta no interrumpe a POETA).

### Manos libres con "poeta" (Fase 4e)

    python main.py voice --wake

Cada enunciado se transcribe en CPU con un Whisper pequeño (`wake_word.stt_model`, por defecto `base`) y solo cuenta si empieza
con "poeta"; lo demás se descarta al instante (no llega al LLM, a la memoria ni a disco). Si dices solo "poeta", suena un tono y
espera tu orden. Limitación: es una aproximación; cualquier frase que empiece con "poeta" activa, y la detección tarda lo que tarda
el STT pequeño. Un modelo openWakeWord propio (más liviano y preciso) queda pendiente.

## Fase 6: herramientas y MCP

El modelo ahora puede usar herramientas (`/tools` las lista). Todas pasan por el Permission Manager y se auditan.

Integradas (activables en `config/tools.yaml`): `get_datetime`, `read_file`, `list_dir`, `write_file` (solo carpeta `workspace/`),
`open_app`, `fetch_url` (bloquea redes privadas) y `run_command` (lista blanca exacta, sin shell).
El contenido que devuelven archivos, paginas y servidores MCP es **dato no confiable**: si aparece en el turno,
las acciones de nivel 3+ piden confirmacion explicita.

### Servidores MCP
1. `uv pip install -e '.[mcp]'`
2. En `config/mcp.yaml`: completar `command`/`args`, `enabled: true` y agregar el nombre a `authorized_servers` (doble llave).
3. `python main.py mcp list` y `python main.py mcp approve NOMBRE`: muestra las herramientas y fija su huella.
   Si el servidor cambia sus herramientas, se desactiva hasta aprobar de nuevo.
4. Por defecto cada herramienta MCP es riesgo `write` (pide confirmacion); bajalo con `risk`/`tool_risk` si confias.
   Al servidor solo llegan las variables listadas en `env`, nunca tus claves.

Limites conocidos: `fetch_url` no cubre DNS rebinding entre la resolucion y la conexion; los servidores MCP de terceros
no estan verificados por POETA; el contenido binario (imagenes) de las herramientas se descarta.

## Fase 5: vision y gestos (5a-5b)

1. `uv pip install -e '.[vision]'` y `python scripts/fetch_models.py` (modelo de manos de MediaPipe, ~7.5 MB).
2. `python scripts/camera_probe.py` identifica tus camaras (color vs monocromo/IR); ajusta `camera.device` en `config/vision.yaml`.
3. En `config/permissions.yaml`, en `actions:`, agrega `camera_capture: 2` (abrir la webcam; auditado).
4. `python scripts/gesture_test.py` muestra la pose en vivo; `--calibrate` ajusta los umbrales de dedos a tu mano.

Arquitectura: `CameraManager` abre la webcam UNA vez y reparte cuadros (cada modulo pide su fps); se libera al irse el ultimo
suscriptor. `HandTracker` (MediaPipe HandLandmarker) -> `GestureEngine` (poses propias por geometria + logica temporal)
-> eventos. Mano abierta 0.6 s = `palm` (detener, siempre activo); 1.5 s = `activate` (activa la interaccion 10 s);
con la interaccion activa: `thumb_up` (confirmar), `thumb_down` (cancelar), `point` (seleccionar), `swipe_left/right`.
Hay espera entre gestos y tolerancia a parpadeos. Un gesto nunca autoriza solo una accion sensible.
Aun no esta conectado al agente ni a la voz (eso es la Fase 9).

## Fase 7: automatizacion del escritorio (KDE / Wayland)

1. `python scripts/add_permission_actions.py` agrega a tu `permissions.yaml` las acciones nuevas (conserva tus ajustes; deja `.bak`).
2. `python scripts/desktop_probe.py` verifica que existan las utilidades: `notify-send`, `playerctl`, `wpctl`, `kdotool`,
   `systemctl --user`, `ps`. Cada herramienta solo se registra si su utilidad esta instalada (`/tools` las lista).
3. Edita `config/desktop.yaml` (tope de volumen, `services.allowlist`) y `config/automations.yaml` (rutinas).

Herramientas: `notify`, `media_status`, `media_control`, `get_volume`, `set_volume`, `list_windows`, `window_action`,
`service_status`, `service_control`, `list_processes`, `stop_process`, `run_automation`.
Niveles por defecto: lectura 1; notificar, multimedia y volumen 2; cerrar/minimizar ventanas, controlar servicios, detener
procesos y ejecutar rutinas 3 (confirmacion). Titulos de ventana, metadatos multimedia y nombres de proceso son datos NO
confiables (el contenido lo controlan terceros). `service_control` solo actua sobre `desktop.services.allowlist`;
`stop_process` solo envia SIGTERM a procesos tuyos y nunca a procesos criticos; `window_action` verifica el titulo real
antes de actuar. Una rutina solo la nombra el modelo: sus pasos son fijos y cada paso pasa ademas por sus propios permisos.
`/run NOMBRE` la ejecuta desde el chat.

## Fase 9: integracion

`python main.py voice --wake --gestures` une voz, herramientas, MCP y gestos en un solo proceso.

- **Estado y metricas:** el bucle publica estados (`idle, listening, transcribing, thinking, tool, speaking, confirming`) en
  un `EventBus` (para el panel/HUD de la Fase 8; nunca llevan texto). Cada turno agrega una fila numerica a
  `logs/metrics.jsonl` (STT, razonamiento, primer texto, primer audio, total, RAM, VRAM). `python main.py metrics [--last N]`
  muestra p50/p95: es la medicion de latencia y de VRAM pendiente en la P16.
- **Gestos en la conversacion:** palma sostenida = callar a POETA. Para confirmar con gestos: mano abierta 1.5 s (activa),
  luego pulgar arriba = autorizar, abajo = cancelar. Solo para confirmaciones de nivel 3; las explicitas (nivel 4, imagenes o
  audio), `cloud_transfer` y las que siguen a datos no confiables (web, ventanas, MCP) se contestan SIEMPRE con el teclado.
  Cada decision por gesto queda auditada (`gesture_confirmation`). Se ajusta en `config/vision.yaml` (gestures:
  `confirm_with_gestures`, `interrupt_with_palm`, ambos true por defecto).
- **Servicio supervisado:** `python scripts/install_service.py [--gestures]` instala `poeta.service` (usuario, con reinicio
  automatico). Corre `voice --wake --headless`: sin teclado, las confirmaciones solo pueden darse por gestos y el resto se
  deniega. `systemctl --user enable --now poeta` y `journalctl --user -u poeta -f`.
