# 0005 — Algoritmos Genéticos: población final y mejor individuo

**Fecha:** 2026-06-24
**Estado:** activo · Lección 4 creada
**Fuente:** `apuntes/CAMPUS VIRTUAL/01 Presentaciones/05 IA - AlgoritmosGeneticos_FIE2024.pdf`
renderizado a PNG y revisado visualmente; parcial de muestra corregido (`consignaParcial.png`,
`respuestasParcial.png`) revisado como imagen.

## Concepto fijado
La tabla de la cátedra define:
- Gen = característica o atributo.
- Cromosoma / genotipo = descripción de la solución.
- Individuo = posible solución.
- Fenotipo = función de aptitud.

## Trampa del parcial
La afirmación 5 ("un gen representa una característica o variable dentro de un cromosoma") es
**VERDADERA**.

La afirmación 6 ("la población final ... se analiza para encontrar soluciones óptimas") es
**FALSA** si se lee literalmente: en teoría la solución se identifica por el **individuo de
mejor aptitud** dentro de la población final, pero en la práctica no se analiza solo esa
población final. Se comparan los logs de varias corridas y puede elegirse el mejor individuo
observado aunque no pertenezca a la población final.

## Evidencia visual
- Diapo 9: equivalencia evolución ↔ AG.
- Diapo 10: flujo Holland 1975.
- Diapo 11: cromosoma como cadena de genes y función de aptitud `f(x): Cromosoma → R`.
- Diapo 24: condiciones aumentan aptitud, restricciones disminuyen, combinaciones inválidas
  penalizan.
- Diapos 67-68: si no hay paro, `Pi = PM`; si hay paro, `PF = PM`.
- Diapo 69: "Identificación del individuo solución (teoría)" por **mejor aptitud** dentro de
  la población final.
- Diapo 75: en práctica se usa log de corridas y mejor aptitud; el individuo elegido puede no
  estar en la población final de una corrida.

## Archivos creados
- `lessons/0004-algoritmos-geneticos.html`
- `reference/algoritmos-geneticos.html`
