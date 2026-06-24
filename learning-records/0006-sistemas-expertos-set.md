# 0006 — Sistemas Expertos: motor de inferencias vs base de conocimientos

## Contexto

Lección 5 del curso de IA para el parcial de la Prof. Vegega. El foco es la afirmación 4
del V/F: métodos de búsqueda en un Sistema Experto Tradicional.

## Aprendizaje

La afirmación del parcial dice que los métodos de búsqueda se aplican en un SET "dentro de
la Base de Conocimientos". La idea evaluada es correcta porque el SET tiene conocimiento
explícito recorrible. Pero la precisión que marcó la profe es que la búsqueda se aplica en
el **motor de inferencias**, que activa reglas usando la base de datos y la memoria de
trabajo.

## Evidencia visual verificada

- `Componentes - Sistemas Expertos Tradicionales.pdf`, páginas 1-2: la base de
  conocimientos se define como unión de aserciones y reglas; el motor de inferencias activa
  reglas y entrega información al trazador de explicaciones.
- `IA-ProcConstrSistInt_FIE2024.pdf`, diapositivas 39-42: en formalización se elige entre
  `SBC / SET`, `AG` y `RNA`. Para clasificación/predicción con reglas completas corresponde
  `SBC / SET`; para ejemplos representativos, `RNA`; para optimización combinatoria, `AG`.
- `respuestasParcial.png`: en la afirmación 4 la corrección manuscrita remarca "aplicar el
  motor de inferencias".

## Regla para justificar en examen

No escribir que la base de conocimientos "hace" la búsqueda. Es depósito de reglas. La
frase fuerte es: la base guarda el conocimiento explícito y el motor de inferencias lo
recorre/aplica mediante métodos de búsqueda.
