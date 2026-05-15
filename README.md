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

## Modelo de seguridad

El adaptador del simulador siempre pasa las predicciones por `SafetyFilter` antes de enviar un comando. El filtro:

- rechaza predicciones NaN o infinitas
- limita velocidades y `yaw_rate`
- suaviza comandos sucesivos
- cambia a hover/stop cuando la profundidad indica un obstaculo cercano

El adaptador de AirSim es opcional y se importa solo cuando se usa. Si `airsim` no esta instalado, los comandos CLI del simulador fallan con un mensaje claro de dependencia en vez de romper los flujos de manifiestos y tests.

## Politica para dron real

Los drones reales quedan fuera del alcance de la v1. Un futuro adaptador MAVSDK/MAVLink/ROS debe mantener estos controles:

- validacion previa en simulador
- override manual
- geocerca
- pruebas de banco
- pruebas de baja velocidad con protectores de helices
- piloto de seguridad
- solo consignas de velocidad de alto nivel

## Verificacion

Chequeos locales que no requieren PyTorch:

```bash
python3 -m pytest
```

Chequeo opcional de sintaxis:

```bash
python3 -m compileall drone_autopilot tests
```
