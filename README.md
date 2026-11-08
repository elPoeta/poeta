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
