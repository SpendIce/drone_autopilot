# 0007 — Lógica de Primer Orden: universal implica, existencial conjuga

## Contexto

Lección 6 del curso de IA para el parcial de la Prof. Vegega. El ejercicio 3 del parcial
pide traducir frases en lenguaje natural a Lógica de Primer Orden.

## Aprendizaje

La regla de alto rendimiento para traducir LPO es identificar el cuantificador principal:

- Si el cuantificador principal es `∀`, la fórmula normalmente contiene implicación `→`.
- Si el cuantificador principal es `∃`, la fórmula no usa implicación como conector central;
  se expresa con conjunción `∧`.

Esto evita el error típico de traducir "algunos A son B" como `∃x(A(x) → B(x))`, que no
afirma la intersección entre A y B.

## Evidencia visual verificada

- `04 IA - Representacion_Conocimiento_Logica_FIE2024.pdf`, diapositivas 7-8: LPO usa
  operadores lógicos, variables, constantes, funciones de verdad/predicados y
  cuantificadores.
- Diapositivas 9-13: ejemplos de serpientes, rosas, personas/círculos y tips explícitos:
  `∀` como cuantificador principal contiene implicación; `∃` como cuantificador principal
  no contiene implicación.
- `02 Guía_Práctica_Lógica.pdf`, segunda sección: el formato del ejercicio pide traducir
  frases a Lógica de Primer Orden.

## Regla para el examen

Primero nombrar predicados y dominio. Después decidir si la frase habla de todos los objetos
de una clase o de la existencia de alguno. Recién ahí escribir conectores. No empezar por los
símbolos.
