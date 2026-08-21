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
# runs/expert_dataset + runs/expert_dataset_v2 subidos como datasets propios:
mkdir -p /kaggle/working/combined_dataset
ln -s /kaggle/input/datasets/lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320/data_collected_potential_final_v58_mod25_320x320_cmds \
  /kaggle/working/combined_dataset/seed
ln -s /kaggle/input/datasets/<tu-usuario>/airsim-blocks-expert-avoidance-rgbd-3k/expert_dataset \
  /kaggle/working/combined_dataset/expert
ln -s /kaggle/input/datasets/<tu-usuario>/<slug-v2>/expert_dataset_v2 \
  /kaggle/working/combined_dataset/expert_v2

python3 -m drone_autopilot.cli build-airsim-manifest /kaggle/working/combined_dataset/seed \
  --source airsim_obstacle_avoidance_seed --path-root /kaggle/working/combined_dataset \
  --output /kaggle/working/seed_manifest.parquet

python3 -m drone_autopilot.cli build-airsim-manifest /kaggle/working/combined_dataset/expert \
  --source expert_avoidance_seed --path-root /kaggle/working/combined_dataset \
  --episode-length 135 --output /kaggle/working/expert_manifest.parquet

python3 -m drone_autopilot.cli build-airsim-manifest /kaggle/working/combined_dataset/expert_v2 \
  --source expert_avoidance_seed_v2 --path-root /kaggle/working/combined_dataset \
  --episode-length 135 --output /kaggle/working/expert_manifest_v2.parquet

python3 -m drone_autopilot.cli merge-manifests \
  /kaggle/working/seed_manifest.parquet /kaggle/working/expert_manifest.parquet \
  /kaggle/working/expert_manifest_v2.parquet \
  --output /kaggle/working/combined_manifest.parquet

torchrun --standalone --nproc_per_node=2 -m drone_autopilot.cli train \
  /kaggle/working/combined_manifest.parquet \
  --data-root /kaggle/working/combined_dataset \
  --backbone mobilenet_v3_small --image-size 224 --epochs 20 --batch-size 32 \
  --distributed \
  --output /kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_avoidance_v2_last.pt \
  --best-output /kaggle/working/rgbd_mobilenet_v3_small_224_e20_ddp_avoidance_v2_best.pt
```

`runs/expert_dataset` pesa ~703MB (241MB rgb, 449MB depth, 13MB commands), `expert_dataset_v2`
~480MB (ya con el episodio degenerado y las colas congeladas descartados, 2102 frames) —
subir los dos como Kaggle Datasets propios antes de linkearlos. Despues del reentrenamiento,
repetir la tabla de la seccion 2 con el checkpoint nuevo para cuantificar la mejora.

### 5. Resultado del reentrenamiento y dos bugs mas encontrados en el loop cerrado

El reentrenamiento (`rgbd_mobilenet_v3_small_224_e20_ddp_avoidance_best.pt`, misma epoca
10 elegida que el checkpoint original) se corrio sobre Kaggle T4x2 con el dataset
combinado (10k semilla + 3105 del experto). Repitiendo la tabla de la seccion 2 con este
checkpoint, en la pared original (`(14,-18) -> (14,-26) -> (36,-26)`):

| Config | Checkpoint | emergency_stops | min_depth_m | distancia restante |
|---|---|---|---|---|
| `blend=0.0` (red pura) | original | 70/135 | 0.24 | 11.03 m |
| `blend=0.0` (red pura) | **reentrenado** | **0/135** | **3.89** | **5.95 m** |
| `blend=0.45/0.5` (config del informe) | original | 0/135 | 2.35 | 8.36 m (no pasa WP0) |
| `blend=0.45/0.5` (config del informe) | **reentrenado** | **0/135** | **6.09** | **pasa WP0**, 19.31 m a WP1 |

La red sola dejo de chocar contra la pared original. Buscando un caso mas duro (un
obstaculo puesto deliberadamente en el medio de una ruta recta, para que "planificador
solo" tuviera que fallar necesariamente) aparecieron tres bugs reales mas en
`SafetyFilter`, todos del mismo patron: el comando ya mezclado con el objetivo del
planificador diluye la senal de evitacion justo cuando mas autoridad hace falta.

**Bug: el freno de emergencia era permanente.** `close_obstacle` devolvia `hover()` sin
condiciones — al no moverse, la profundidad no cambia, con lo cual el freno se
retrigger indefinidamente aunque la red intente esquivar. Fix en `safety.py`
(`_escape_command`): mientras haya un obstaculo cerca, se prohibe seguir acercandose
(`vx <= 0`, freno o retroceso, nunca acelerar hacia el obstaculo) pero se deja pasar el
movimiento lateral/yaw. Test: `test_safety_filter_close_obstacle_forbids_forward_approach`,
`test_safety_filter_close_obstacle_permits_backing_away`.

**Bug: el blend del planificador diluia la senal de esquive.** Aun con el fix anterior,
`SafetyFilter.filter()` solo recibia el comando ya mezclado con el objetivo
(`MissionPlanner` blend), y esa mezcla le recortaba a la mitad el `yaw_rate` que la red
pedia (medido: `predicted_yaw_rate` promedio 0.384 rad/s vs `planned_yaw_rate` 0.152 en el
tramo trabado). `filter()` ahora acepta un parametro `reactive` opcional con la prediccion
cruda pre-blend, y lo usa para `vy`/`yaw_rate` durante la emergencia en vez del comando
diluido (`vx` sigue gobernado por el comando real, no por el crudo). Wireado en
`run_closed_loop` y `record_episode`. Test:
`test_safety_filter_close_obstacle_steers_by_raw_reactive_command`.

**Bug: el mismo blend diluia tambien el retroceso (`vx`).** Con los dos fixes anteriores,
sobre la ruta con obstaculo forzado (`TemplateCube_Rounded_77`) el dron seguia quedando
fisicamente atascado — posicion y yaw exactamente congelados durante cientos de pasos, con
`command_yaw_rate≈-0.4` sostenido sin efecto. Parecia un "local minimum" estructural de la
navegacion reactiva (se probo `emergency_depth_m` en 0.8/1.2/1.6/2.0/2.4m sin cambios), pero
resulto ser el mismo bug de dilucion, esta vez en `vx`: `_escape_command` solo vetaba avance
usando el `vx` ya mezclado con el objetivo (`command`), y ese blend podia cancelar un
retroceso real de la politica cruda (`vx` negativo mezclado con el `vx` positivo del
objetivo da un valor cercano a cero, que el veto de "no avanzar" no distingue de "ya frenado
del todo"). Fix: la velocidad de escape toma `min(command.vx, steering_source.vx, 0.0)` — si
cualquiera de las dos fuentes quiere retroceder, retrocede; el objetivo ya no puede pisar un
retroceso activo con su tironeo hacia adelante. Test:
`test_safety_filter_close_obstacle_honors_raw_retreat_over_diluted_blend`.

Con los tres fixes, el mismo experto que antes quedaba congelado con `emergency_stops=142/200`
en un intento de la ruta obstruida (200 pasos) bajo a **6/200**, y la posicion dejo de
congelarse esa vez: paso el punto donde antes se trababa (min_depth bajo a 0.78m y se
recupero a 1.63m en 20 pasos). `expert_policy.py` ademas gira mas temprano y fuerte
(`caution_depth_m=4.5`, `urgency_exponent<1` para front-load el giro) para que el retroceso
sea el ultimo recurso, no la estrategia principal.

**Pero no es 100% determinista.** Una corrida mas larga (500 pasos), mismos parametros
exactos, volvio a quedar congelada — esta vez con retroceso real activo
(`vx=-0.268`, no diluido) y las tres variables al maximo, sin mover un milimetro durante
400+ pasos. La diferencia con la corrida que si zafo parece ser el timing exacto de
contacto: una vez que la fisica de UE4/AirSim registra penetracion real de colision
(`state_collided=True`, confirmado en una corrida anterior), ningun comando de velocidad
la saca — es una propiedad de esta esquina concava puntual, no de los parametros. Los tres
fixes son mejoras reales y generalizables (verificadas con tests unitarios y con la mejora
de 70→0 en la pared original), pero no garantizan clarear esta geometria especifica al
100% — por eso el foco paso a grabar una campaña de demostraciones con el experto
actualizado (gira temprano, retrocede como ultimo recurso) en vez de seguir persiguiendo
un unico obstaculo puntual.

**Detector de atascamiento (memoria minima, tipo Bug1/Bug2).** `_escape_command` reacciona
proporcionalmente cada paso — alcanza para la mayoria de los obstaculos, pero en la esquina
concava la profundidad quedaba exactamente congelada durante cientos de pasos con la señal
en autoridad maxima: reaccionar igual cada paso nunca escala. `SafetyFilter` ahora cuenta
pasos consecutivos en `close_obstacle` sin ganancia real de profundidad
(`stuck_streak_threshold`, default 15, con `stuck_depth_improvement_m=0.05` como umbral de
"mejoro de verdad"); al cruzar el umbral, en vez de seguir re-decidiendo paso a paso, se
compromete a una maniobra fija y sostenida (`stuck_escape_steps` pasos de reversa +
lateral/yaw a tope, sin suavizado) en la direccion detectada al momento del commit. Tests:
`test_safety_filter_commits_to_stuck_escape_after_streak_of_frozen_depth`,
`test_safety_filter_stuck_escape_lasts_configured_duration_then_resumes_reactive`,
`test_safety_filter_stuck_streak_resets_when_depth_improves`.

Probado en vivo sobre la misma esquina: el mecanismo se activa y alterna correctamente
entre respuesta proporcional y commit fijo (se ve en la telemetria: `vx=-0.6, vy=-0.6,
yaw=-34.4°/s` exactos en los pasos de commit), pero `min_depth` sigue sin moverse de
0.70-0.72m durante 150+ pasos. Con tres estrategias de control distintas fallando en el
mismo punto exacto, la conclusion fue que esta esquina puntual involucra penetracion fisica
real de la malla de colision de AirSim/UE4 — hasta que el fix de mas abajo demostro que no
era asi.

**El fix real: destemplar el blend del planificador por proximidad, no solo en el limite
de emergencia.** El "diluye la señal de esquive" que motivo los dos fixes anteriores solo se
habia corregido *dentro* de la banda de emergencia (`SafetyFilter`); durante el acercamiento
—antes de cruzar ese umbral, pero ya con un obstaculo cerca— el blend seguia constante,
tironeando hacia el objetivo con la misma fuerza sin importar la distancia. `MissionPlanner.
update()` ahora acepta `avoidance_urgency` (0..1) y reduce `position_blend`/`yaw_blend` en
esa proporcion; `SafetyFilter.urgency_for_depth()` calcula la señal a partir de una nueva
banda de precaucion (`caution_depth_m`, sin estado, se puede llamar antes del blend). Con
esto, el giro de esquive llega con autoridad completa durante todo el acercamiento, no solo
en el ultimo instante.

Probado en vivo sobre la misma esquina que fallaba con las tres estrategias anteriores:
**0/250 emergency stops de punta a punta**. La profundidad toca 1.23m exactamente en el
punto donde antes quedaba atascada, y se recupera a 2.43m en vez de congelarse — sigue
esquivando y recuperandose varias veces mas a lo largo del corredor (otros obstaculos en la
ruta de 34m), terminando en espacio abierto (min_depth 6.56m). La conclusion sobre
"penetracion fisica" del parrafo anterior era prematura: el problema seguia siendo dilucion
de señal, esta vez en la fase de acercamiento en vez de en la emergencia misma. Este fix
funciona con el checkpoint ya entrenado — no depende de reentrenar. Tests:
`test_mission_planner_avoidance_urgency_tapers_goal_contribution`,
`test_urgency_for_depth_ramps_linearly_between_bands`.

Validado en 3 rutas mas antes de tratarlo como probado (una corrida limpia sola podia ser
suerte): la pared original (chequeo de regresion, `0/135`), la misma zona del obstaculo con
offset lateral por si era mas ancho de lo pensado (`0/200`), y un acercamiento norte-sur al
mismo punto desde un angulo nunca probado (`0/200`, `mission_complete=True`). Las tres
corridas, cero emergency stops.

**Segunda campaña de grabacion.** `scripts/record_expert_campaign_v2.py`: misma grilla de
posiciones/rumbos que la primera (16 episodios, sin la diagonal para acotar tiempo) mas 3
repeticiones de la ruta obstruida, con `depth-interval=2` (en vez de 5) y los defaults
nuevos de `expert_policy.py`. Resultado: 2910 frames; 15/16 episodios de grilla con
`mission_complete=True` y 0 emergency stops; 1 episodio degenerado (mismo spawn-dentro-de-
obstaculo de `(22,-14)` de la campaña anterior); las 3 repeticiones de la ruta obstruida
quedaron atascadas (~160-166/250 cada una, consistente con el hallazgo de arriba). Antes de
armar el manifest, descartar el episodio degenerado y — a diferencia de la primera campaña —
evaluar si conviene recortar la cola congelada de los 3 intentos obstruidos (frames
duplicados con "retroceder a tope" como etiqueta repetida cientos de veces sesgaria el
dataset) en vez de descartar el episodio completo, ya que la primera mitad de esos 3
intentos (el acercamiento y el giro temprano, antes de trabarse) es señal util que no
esta en la primera campaña.

### 6. Reentrenamiento v2 (seed + experto v1 + experto v2) y resultado de la red sola

Reentrenado en Kaggle T4x2 con las tres fuentes combinadas (`merge-manifests` con seed +
`expert_avoidance_seed` + `expert_avoidance_seed_v2`,
`rgbd_mobilenet_v3_small_224_e20_ddp_avoidance_v2_best.pt`). Con `blend=0.0` (red pura, sin
ninguna ayuda del planificador ni del fix de `avoidance_urgency` — el blend ya es 0 asi que
no hay nada que destemplar):

| Ruta | emergency_stops | min_depth_m | resultado |
|---|---|---|---|
| Pared original (135 pasos) | 0/135 | 2.41 | avanza, 6.17m restantes |
| Ruta obstruida (200 pasos) | 0/200 | 2.01 | avanza, 18.15m restantes (sin objetivo que perseguir) |

La red **sola** —sin el fix de `MissionPlanner`, sin el detector de atascamiento, sin nada
del lado del `SafetyFilter` mas alla del freno de emergencia basico— ya esquiva ambas rutas
sin un solo emergency stop. Confirma que las demostraciones de las dos campañas (giro
temprano, retroceso de ultimo recurso) se transfirieron al comportamiento aprendido, no
solo al comportamiento del experto que las genero.

### 7. Ablacion RGB vs. Depth vs. RGB-D — ejecutada, con un sesgo de dataset a declarar

Corrida sobre el dataset combinado (seed + experto v1 + experto v2), mismos hiperparametros
que las corridas de la Tabla 2, solo cambia `--modality`. Metricas extraidas directamente de
`extra.history` dentro de cada checkpoint (no hace falta reentrenar para volver a verlas):

| Modalidad | val_mae_vx | val_mae_vz | val_mae_yaw_rate | val_score (normalizado) | mejor epoca |
|---|---|---|---|---|---|
| RGB solo | 0.486 | 0.177 | 0.149 | 1.604 | 7 |
| Depth solo | 0.267 | 0.114 | 0.088 | 0.969 | 19 |
| RGB-D completo | 0.282 | 0.111 | 0.089 | 0.973 | 19 |

Depth solo empata (marginalmente mejor) con RGB-D completo; los dos superan claramente a RGB
solo. **Esto no significa que RGB sea irrelevante en general — es un sesgo de como se
construyo el dataset.** `ReactiveAvoidancePolicy` (el experto de las campañas 1 y 2) calcula
sus acciones exclusivamente a partir de profundidad, sin usar RGB. Como esos episodios son
~5207 de los ~15207 frames del dataset combinado (~34%), un tercio de las etiquetas de
entrenamiento son, por construccion, una funcion determinista de la profundidad sola —
eso favorece a `depth` en la ablacion independientemente de si RGB aportaria algo en un
dataset generado de otra forma (ej. demostraciones humanas, donde el piloto si mira la
imagen a color). Se declara como limitacion metodologica de esta ablacion especifica, no
como conclusion general sobre la arquitectura RGB-D.

**Velocidad: escalado seguro validado.** En la pared original, subir `mission-cruise-speed`
y `max-vx` junto con `emergency-depth` en la misma proporcion (para no perder margen de
frenado) funciona limpio hasta 3x la velocidad original:

| Velocidad crucero | max-vx | emergency-depth | emergency_stops | resultado |
|---|---|---|---|---|
| 0.5 m/s (original) | 0.50 | 0.8 | 0/135 | avanza, no completa en 135 pasos |
| 1.0 m/s (2x) | 1.0 | 1.6 | 0/135 | pasa WP0, 6.10m restantes a WP1 |
| 1.5 m/s (3x) | 1.5 | 2.4 | 0/135 | **`mission_complete: true`** |

A 3x, la mision completa las dos etapas en los mismos 135 pasos que a velocidad original
apenas avanzaban. La regla es la esperada: subir velocidad sin subir el margen de
frenado en proporcion reintroduce el riesgo de colision (menor distancia de reaccion
para la misma profundidad de deteccion); subiendolos juntos, no.

**Performance del loop: `depth-interval` es un trade-off de throughput vs. frescura, no
solo un ahorro de computo.** Medido en corridas equivalentes (135 pasos, mismo mapa):

| `--depth-interval` | `mean_capture_s` | `mean_step_hz` | Resultado en la ruta obstruida |
|---|---|---|---|
| 5 (config original del informe) | ~0.17-0.22s | ~1.4-3.7 Hz | Colision fisica real (`state_collided=True`) — la profundidad llega hasta 3.2s desactualizada |
| 1 (maxima frescura) | ~0.5-0.7s | ~0.6-0.8 Hz | Sin colision, pero el loop se vuelve ~5x mas lento |
| 2 (usado en la campaña v2) | ~0.3s | ~3.1 Hz | Sin colision, throughput cercano al original |

`depth-interval=5` fue el valor "verificado" del informe original porque el checkpoint viejo
nunca se acercaba lo suficiente a un obstaculo para que la latencia de sensado importara. Con
el piloto reentrenado navegando mas cerca de los obstaculos, la misma configuracion deja de
ser segura — la profundidad reportada puede estar hasta `depth-interval * command-duration`
segundos vieja (3.2s en el caso original), tiempo mas que suficiente para recorrer el margen
de emergencia completo a velocidad de crucero. `depth-interval=2` es el punto medio recomendado
para las corridas de este proyecto: recupera casi todo el throughput de `=5` sin la latencia de
sensado que causo la colision.

## Verificacion

Chequeos locales que no requieren PyTorch:

```bash
python3 -m pytest
```

Chequeo opcional de sintaxis:

```bash
python3 -m compileall drone_autopilot tests
```
