# Piloto de dron RGB-D orientado primero a simulacion

Este proyecto es una base en Python para un piloto reactivo de dron:

`RGB + profundidad -> comando seguro de velocidad en el marco del cuerpo`

El primer objetivo es la simulacion de lazo cerrado. El uso en un dron real queda intencionalmente bloqueado por una lista de seguridad y solo deberia enviar consignas de velocidad de alto nivel mediante un adaptador futuro.

## Estrategia de datasets

El dataset local de AirSim con RGB/profundidad/comandos se trata como dataset semilla supervisado porque tiene etiquetas exactas `[vx, vy, vz, yaw_rate]`. El generador de manifiestos convierte `yaw_rate` de grados por segundo a radianes por segundo y guarda los comandos en el estandar interno:

- `vx`, `vy`, `vz`: metros por segundo
- `yaw_rate`: radianes por segundo
- `action_frame`: `body`

Mid-Air y DDOS quedan previstos como fuentes posteriores de manifiestos: Mid-Air puede aportar pseudoacciones derivadas de diferencias de pose, mientras que DDOS conviene mas para robustez de profundidad/percepcion o tareas auxiliares.

## Inicio rapido

Ejecutar los comandos desde la raiz del proyecto, no desde dentro de `drone_autopilot/`:

```bash
cd /home/spendice/Documents/Archivos_Facu/IA
```

Construir un manifiesto para el dataset semilla local de AirSim:

```bash
python3 -m drone_autopilot.cli build-airsim-manifest \
  data_collected_potential_final_v58_mod25_320x320_cmds
```

Inspeccionar estadisticas de acciones:

```bash
python3 -m drone_autopilot.cli stats \
  data_collected_potential_final_v58_mod25_320x320_cmds/manifest.parquet
```

Instalar las dependencias de entrenamiento antes de entrenar el modelo:

```bash
python3 -m pip install -e ".[training,dev]"
```

Entrenar un piloto RGB-D chico:

```bash
python3 -m drone_autopilot.cli train \
  data_collected_potential_final_v58_mod25_320x320_cmds/manifest.parquet \
  --data-root data_collected_potential_final_v58_mod25_320x320_cmds \
  --epochs 10 \
  --backbone mobilenet_v3_small
```

El codigo de entrenamiento enmascara etiquetas no disponibles, normaliza acciones usando solo estadisticas del split de entrenamiento y reporta metricas por salida y por fuente.

### Entrenamiento multi-GPU

El modo multi-GPU recomendado usa `DistributedDataParallel` mediante `torchrun`. No uses `python -m ... --multi-gpu` como camino principal: ese modo mantiene `DataParallel` solo para pruebas, pero puede fallar en Kaggle T4 x2 con errores cuDNN.

En Kaggle T4 x2, lanzar dos procesos, uno por GPU:

```bash
torchrun --standalone --nproc_per_node=2 -m drone_autopilot.cli train \
  /kaggle/working/manifest.parquet \
  --data-root /kaggle/input/datasets/lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320/data_collected_potential_final_v58_mod25_320x320_cmds \
  --backbone mobilenet_v3_small \
  --image-size 224 \
  --epochs 20 \
  --batch-size 32 \
  --distributed \
  --output /kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_last.pt \
  --best-output /kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_best.pt
```

En modo distribuido, `--batch-size` es por GPU. Con `--nproc_per_node=2 --batch-size 32`, el batch efectivo global es 64.

El trainer guarda el ultimo checkpoint en `--output` y el mejor checkpoint de validacion en `--best-output`. La seleccion del mejor modelo usa `val_mae_vx + val_mae_vz + val_mae_yaw_rate`; `vy` queda fuera porque en el dataset semilla suele ser casi constante.

## Modelo de seguridad

El adaptador del simulador siempre pasa las predicciones por `SafetyFilter` antes de enviar un comando. El filtro:

- rechaza predicciones NaN o infinitas
- limita velocidades y `yaw_rate`
- suaviza comandos sucesivos
- cambia a hover/stop cuando la profundidad indica un obstaculo cercano

El adaptador de AirSim es opcional y se importa solo cuando se usa. Si `airsim` no esta instalado, los comandos CLI del simulador fallan con un mensaje claro de dependencia en vez de romper los flujos de manifiestos y tests.

### Dependencia AirSim

El adaptador actual usa la API clasica de AirSim OSS: `airsim.MultirotorClient`, `simGetImages` y `moveByVelocityBodyFrameAsync`. Project AirSim no es retrocompatible como reemplazo directo: usa cliente `projectairsim`, objetos `World`/`Drone`, configuracion JSONC y metodos async con `asyncio`. Para Project AirSim haria falta un adaptador nuevo.

En Python moderno, el paquete clasico `airsim==1.8.1` requiere instalar algunas dependencias legacy antes y desactivar build isolation:

```bash
python -m pip install msgpack-rpc-python backports.ssl_match_hostname
python -m pip install --no-build-isolation "airsim==1.8.1"
```

Verificar cliente:

```bash
python -c "import airsim; print(airsim.MultirotorClient, airsim.ImageRequest, airsim.YawMode)"
```

## Politica para dron real

Los drones reales quedan fuera del alcance de la v1. Un futuro adaptador MAVSDK/MAVLink/ROS debe mantener estos controles:

- validacion previa en simulador
- override manual
- geocerca
- pruebas de banco
- pruebas de baja velocidad con protectores de helices
- piloto de seguridad
- solo consignas de velocidad de alto nivel

## Experimentos para el informe (limitaciones declaradas en la Discusion)

El informe reconoce tres limitaciones metodologicas en la seccion de Discusion. La
infraestructura para resolver las tres ya existia en el codigo; los puntos 2 y 3 ya se
ejecutaron y dieron un hallazgo mas fuerte que la limitacion original declarada (ver
seccion 4). El punto 1 sigue pendiente porque requiere GPU/Kaggle.

### 1. Ablacion RGB vs. Depth vs. RGB-D

`--modality` ya soporta las tres variantes (`cli.py`). Correr los mismos hiperparametros
del entrenamiento verificado (Tabla 2) cambiando solo la modalidad, y comparar
`val_mae_vx`, `val_mae_vz`, `val_mae_yaw_rate` de cada corrida:

```bash
for modality in rgb depth rgbd; do
  torchrun --standalone --nproc_per_node=2 -m drone_autopilot.cli train \
    /kaggle/working/manifest.parquet \
    --data-root /kaggle/input/datasets/lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320/data_collected_potential_final_v58_mod25_320x320_cmds \
    --backbone mobilenet_v3_small \
    --modality "$modality" \
    --image-size 224 \
    --epochs 20 \
    --batch-size 32 \
    --distributed \
    --output "/kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_${modality}_last.pt" \
    --best-output "/kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_${modality}_best.pt"
done
```

Con `rgb` y `depth`, el encoder de la rama ausente no se ejecuta (una sola rama, no dos
concatenadas), asi que el MAE resultante cuantifica el aporte real de cada modalidad
por separado.

### 2. Aislar el aporte de la red vs. el planificador determinista — ejecutado

`MissionConfig.position_blend` mezcla comando reactivo y guia determinista via
`(1 - weight) * reactivo + weight * objetivo` (`mission.py:_blend`). Es decir:
**`position_blend=0.0` es la red pura** (el objetivo del waypoint no pesa nada) y
**`position_blend=1.0` es el planificador puro** (la prediccion de la red no pesa nada).
Corridas reales sobre `rgbd_mobilenet_v3_small_224_e20_ddp_best.pt`, mismo punto de
partida `(14,-18)` y waypoints `(14,-26) -> (36,-26)`:

| Config | emergency_stops | min_depth_m | avanzo al siguiente waypoint | distancia restante |
|---|---|---|---|---|
| `blend=0.0` (red pura) | 70/135 | 0.24 | no | 11.03 m |
| `blend=1.0` (planificador puro) | 0/135 | 2.96 | si | 12.36 m (al WP1) |
| `blend=0.45/0.5` (config del informe) | 0/135 | 2.35 | no | 8.36 m |

Hallazgo (ver seccion 4 para el mecanismo): la red sola choco contra una pared y quedo
trabada; el planificador solo no choco simplemente porque su ruta geometrica no cruzaba
ningun obstaculo en esta corrida, no porque "vea" profundidad — el planificador no usa
vision para nada. Lo que evita el choque en las tres configuraciones es el mismo freno
determinista de `SafetyFilter` (hover ciego bajo el umbral de profundidad), no una
decision de rumbo aprendida.

Repetir con mas rutas/obstaculos antes de reportar esto como generalizable (la Discusion
ya pide "repeticiones controladas").

### 3. Metrica de seleccion con unidades heterogeneas — ya corregido

`_validation_score` en `training.py` ahora acepta una normalizacion opcional: divide cada
`MAE` por el desvio estandar de esa accion en el split de entrenamiento (`ActionStats`,
ya calculado) antes de aplicar el peso, y `train()` la usa por defecto. Esto convierte la
suma de `m/s + m/s + rad/s` en una suma de errores relativos adimensionales, sin cambiar
la firma publica del entrenamiento. Reentrenar (o re-evaluar los checkpoints existentes)
y verificar si el checkpoint elegido por `val_score` normalizado sigue siendo la epoca 10.

### 4. La evitacion de obstaculos no es una decision de rumbo aprendida — causa raiz y fix en curso

`SafetyFilter.filter()` (`safety.py:66-73`), ante un obstaculo cercano, no gira ni
retrocede: devuelve `VelocityCommand.hover()`. Es un freno ciego, no una maniobra. La red
tampoco aprendio a esquivar: se entrenamiento por regresion supervisada offline
(Smooth L1 enmascarada) imitando el dataset semilla, sin objetivo ni penalizacion de
colision. El resultado (seccion 2): en un mapa nuevo (AirSim Blocks, `FlyingExampleMap`,
distinto del escenario de origen del dataset), la red sola condujo derecho contra una
pared y quedo trabada en el freno de emergencia.

**Fix en curso — imitacion con demostraciones de evitacion real.** Se agrego
`drone_autopilot/expert_policy.py` (`ReactiveAvoidancePolicy`): controlador determinista
que solo usa profundidad (sin red) — divide la ROI de profundidad en mitad izquierda y
derecha, y cuando el minimo central entra en la banda de precaucion, reduce `vx` y
empuja `vy`/`yaw_rate` en direccion contraria al lado mas cercano; en zona despejada
cruza recto. Este experto **no reemplaza a la red en produccion**: se usa para grabar
demostraciones nuevas via el subcomando `record-expert` (`drone_autopilot/record.py`),
que reutiliza `MissionPlanner`/`SafetyFilter` y escribe frames en el layout que
`build-airsim-manifest` ya consume (`rgb/<frame>.png`, `depth/<frame>.npy`,
`commands/<frame>.npy`).

Campaña ejecutada: 24 episodios (`scripts/record_expert_campaign.py`) en una grilla de
8 posiciones de partida x 3 rumbos dentro de `FlyingExampleMap`, yaw inicial apuntado
hacia el waypoint. Resultado: 3375 frames grabados; 2 episodios (270 frames) se
descartaron porque el punto de partida `(22,-14)` caia dentro del radio de emergencia y
el dron quedo en hover ciego los 135 pasos sin moverse (frames duplicados, sin senal util
de evitacion). Dataset final: **3105 frames**, todos los episodios restantes con
`emergency_stops=0` y `mission_complete=true` — el experto llega al waypoint sin chocar
en cada ruta valida probada.

```bash
python3 -m drone_autopilot.cli record-expert \
  --output-dir runs/expert_dataset \
  --steps 135 --command-duration 0.8 \
  --max-vx 0.50 --max-vy 0.40 --max-vz 0.0 --max-yaw-rate 0.5 \
  --smoothing-alpha 0.12 --emergency-depth 0.8 --depth-roi-bottom 0.55 \
  --arm --takeoff --takeoff-altitude 3.0 --hold-altitude --async-commands \
  --depth-interval 5 --keep-api-control \
  --start-x 14.0 --start-y -18.0 --start-z -3.0 --start-yaw-deg 0.0 \
  --waypoint 14.0,-26.0 --waypoint 36.0,-26.0 \
  --mission-cruise-speed 0.50 --mission-waypoint-radius 2.5 \
  --mission-position-blend 0.6 --mission-yaw-blend 0.6 --mission-max-yaw-rate 0.25 \
  --cruise-vx 0.6 --caution-depth 3.0 --min-forward-depth 1.2 \
  --max-lateral-vy 0.5 --max-avoid-yaw-rate 0.4
```

Manifest del dataset del experto (usar `--episode-length` igual a los `--steps` de cada
episodio para que el split train/val/test respete limites de trayectoria real):

```bash
python3 -m drone_autopilot.cli build-airsim-manifest runs/expert_dataset \
  --source expert_avoidance_seed --episode-length 135
```

**Combinar con el dataset semilla de Kaggle y reentrenar.** `train()` solo acepta un
`--data-root`, asi que ambos datasets deben quedar bajo una raiz comun antes de mezclar
los manifests (`merge-manifests`, agregado en `manifest.py`/`cli.py`):

```bash
# En Kaggle, con el dataset semilla ya montado en /kaggle/input/... y
# runs/expert_dataset subido como dataset propio (o copiado a /kaggle/working/):
mkdir -p /kaggle/working/combined_dataset
ln -s /kaggle/input/datasets/lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320/data_collected_potential_final_v58_mod25_320x320_cmds \
  /kaggle/working/combined_dataset/seed
ln -s /kaggle/input/datasets/<tu-usuario>/expert-avoidance-dataset/expert_dataset \
  /kaggle/working/combined_dataset/expert

python3 -m drone_autopilot.cli build-airsim-manifest /kaggle/working/combined_dataset/seed \
  --source airsim_obstacle_avoidance_seed --path-root /kaggle/working/combined_dataset \
  --output /kaggle/working/seed_manifest.parquet

python3 -m drone_autopilot.cli build-airsim-manifest /kaggle/working/combined_dataset/expert \
  --source expert_avoidance_seed --path-root /kaggle/working/combined_dataset \
  --episode-length 135 --output /kaggle/working/expert_manifest.parquet

python3 -m drone_autopilot.cli merge-manifests \
  /kaggle/working/seed_manifest.parquet /kaggle/working/expert_manifest.parquet \
  --output /kaggle/working/combined_manifest.parquet

torchrun --standalone --nproc_per_node=2 -m drone_autopilot.cli train \
  /kaggle/working/combined_manifest.parquet \
  --data-root /kaggle/working/combined_dataset \
  --backbone mobilenet_v3_small --image-size 224 --epochs 20 --batch-size 32 \
  --distributed \
  --output /kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_avoidance_last.pt \
  --best-output /kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_avoidance_best.pt
```

`runs/expert_dataset` pesa ~703MB (241MB rgb, 449MB depth, 13MB commands) — subirlo como
Kaggle Dataset propio (CLI `kaggle datasets create` o la UI) antes de linkearlo. Despues
del reentrenamiento, repetir la tabla de la seccion 2 con el checkpoint nuevo para
cuantificar la mejora.

## Verificacion

Chequeos locales que no requieren PyTorch:

```bash
python3 -m pytest
```

Chequeo opcional de sintaxis:

```bash
python3 -m compileall drone_autopilot tests
```
