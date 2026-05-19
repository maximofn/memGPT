# ¿Qué hace exactamente MemGPT?

MemGPT es un sistema que le da a un LLM con ventana de contexto **finita** (p. ej. 8k tokens) la **ilusión de tener memoria infinita**, copiando la idea de la *memoria virtual* de los sistemas operativos: igual que un SO pagina datos entre RAM y disco, MemGPT pagina información entre el contexto del LLM y un almacenamiento externo.

---

## 1. La idea central: el LLM como "CPU" de un SO

El LLM ya no es la aplicación entera - es solo el procesador. MemGPT envuelve a ese procesador con:

- Una **jerarquía de memoria** (rápida pero pequeña vs. lenta pero infinita).
- Un **gestor de control de flujo** que decide cuándo invocar al LLM.
- Un conjunto de **funciones** que el propio LLM llama para administrar su memoria.

Lo crítico: **el LLM es quien decide** qué guardar, qué buscar y qué descartar. No es un humano ni un sistema externo - es auto-dirigido.

### Cómo se ve la ventana de contexto

En un agente común, la ventana de contexto suele ser simplemente:

```
Ventana: [
  system prompt;
  [(user) Hola;
   (agent) Hola, qué necesitas?;
   (user) Tengo una duda;
   (agent) Sobre qué?]
]
```

En MemGPT, los *prompt tokens* se dividen en **tres secciones contiguas** (sección 2.1 del paper):

1. **System Instructions** - el system prompt de MemGPT, que ya incluye dentro la descripción de la jerarquía de memoria y los esquemas/descripciones de las funciones (no son bloques separados).
2. **Working Context** - bloque reservado siempre, aunque no se haya llamado ninguna función. Al inicio está vacío o contiene la assistant inicial del agente.
3. **FIFO Queue** - cuyo primer índice está reservado para un *resumen recursivo* de los mensajes eviccionados (expulsados) (vacío al principio).

Para el mismo ejemplo de conversación, la ventana en MemGPT quedaría:

```
Ventana: [
  System Instructions (prompt de MemGPT + descripción de memoria + schemas de funciones);
  Working Context (vacío o assistant inicial);
  FIFO Queue [
    (resumen recursivo, vacío);
    (user) Hola;
    (agent) Hola, qué necesitas?;
    (user) Tengo una duda;
    (agent) Sobre qué?
  ]
]
```

### Qué es el Working Context

El **Working Context** es la "libreta" del agente: vive **dentro de la ventana de contexto** (entre las System Instructions y la FIFO Queue) y guarda lo que el agente considera lo bastante importante como para tenerlo **siempre delante**, en cada inferencia.

#### Estructura: bloques etiquetados

En el paper original, el Working Context era **un único bloque de texto libre**. La implementación de referencia (Letta) lo refinó dividiéndolo en **varios bloques separados, cada uno con su etiqueta y su propio presupuesto de tokens**:

| Etiqueta | Qué guarda | Ejemplo |
|---|---|---|
| `assistant` | Identidad del agente: nombre, personalidad, rol, restricciones. | *"Soy Sam, un asistente paciente y curioso. No revelo que soy una IA."* |
| `human` | Información sobre el usuario que el agente recuerda. | *"El usuario se llama Ana, vive en Madrid, le gusta el surf."* |
| `custom` (cualquier otra) | Bloques adicionales definidos por el desarrollador. | `task_context`, `team_members`, `project_constraints`... |

**Por qué este cambio mejora al diseño del paper**:

- El agente puede **modificar un bloque sin tocar los otros** (`core_memory_replace(label="human", ...)` no afecta a `assistant`).
- Cada bloque tiene **su propio límite de tokens** - más fácil prevenir que la información del usuario "ahogue" la assistant del agente, o viceversa.
- Es **más estructurado** y permite **reusar la misma assistant en múltiples agentes** (factorización).

Las funciones de modificación son las mismas conceptualmente, pero ahora reciben la etiqueta del bloque:

```
core_memory_append(label="human", content="Birthday is February 7")
core_memory_replace(label="human", old="Boyfriend named James", new="Ex-boyfriend named James")
```

#### Propiedades clave

| Propiedad | Explicación |
|---|---|
| **Tamaño fijo por bloque** | Cada bloque tiene su presupuesto de tokens - no puede crecer indefinidamente. |
| **Read/Write** | A diferencia de las System Instructions (read-only), el agente puede modificar cualquier bloque. |
| **Escritura solo vía funciones** | El LLM no escribe directamente generando texto; usa funciones tipo `core_memory_append(label, content)` o `core_memory_replace(label, old, new)`. |
| **Texto no estructurado dentro del bloque** | El contenido de cada bloque es texto libre en lenguaje natural, no JSON ni esquemas rígidos. |
| **Persistente dentro de la ventana** | A diferencia de los mensajes de la FIFO Queue, **nunca se evicta (expulsa) automáticamente**. Sigue ahí hasta que el agente decida cambiarlo. |
| **Persistente entre sesiones** | Se serializa a base de datos. Al reabrir una sesión nueva días después, todos los bloques se recargan tal y como quedaron. |

Es el **único bloque del contexto** que combina dos propiedades a la vez: está siempre visible al LLM **y** sobrevive a las evicciones (expulsiones). Por eso es donde el agente concentra el "mapa mental" del usuario y de sí mismo (datos clave, preferencias, assistant del agente, estado evolutivo de la relación).

#### Cuándo escribe el agente en él

1. Ve algo importante en la conversación y decide consolidarlo.
2. Recibe una alerta de *Memory Pressure* (Figura 1) y rescata lo importante de la FIFO antes de que se evicte (expulse).

Si Recall/Archival (evocación/documental) son **información que el agente tiene que ir a buscar activamente** (y puede fallar al hacerlo), el Working Context es **información que el agente ve siempre, sin necesidad de buscarla**.

### Qué es la FIFO Queue

La **FIFO Queue** (cola FIFO, *First In First Out*) es la **memoria de trabajo conversacional** del agente: un buffer rodante donde se acumula todo lo que va pasando (mensajes, llamadas a funciones, resultados, alertas) en orden cronológico, y del que se van expulsando los elementos más antiguos cuando se llena.

Sus propiedades clave:

| Propiedad | Explicación |
|---|---|
| **Tamaño máximo dinámico** | No es de tamaño fijo absoluto, pero está acotado: lo que sobra de la ventana de contexto tras restar System Instructions y Working Context. |
| **Read/Write vía Queue Manager** | El LLM no escribe directamente en ella generando texto: es el Queue Manager quien añade entradas automáticamente. |
| **Orden cronológico** | Los elementos más antiguos están al principio, los más nuevos al final. Cuando se llena, se expulsan por el principio (FIFO). |
| **Slot 0 reservado** | El primer índice contiene siempre un *resumen recursivo* de los mensajes ya expulsados - es el "qué pasó antes" condensado por el LLM. |
| **Volátil pero recuperable** | Lo que se expulsa de la FIFO no se pierde: queda guardado en Recall Storage (evocación) y puede traerse de vuelta con `recall_storage.search(...)`. |

### Qué se guarda en la FIFO

A diferencia del modelo simple de "solo mensajes user/agent", la FIFO en MemGPT guarda **toda la traza de la interacción**:

- **Mensajes del usuario**: lo que escribe el humano.
- **Mensajes del agente al usuario**: técnicamente son llamadas a `send_message(...)` - en MemGPT toda salida del LLM es una function call, no hay "respuesta directa".
- **Llamadas a funciones de MemGPT**: `working_context.append(...)`, `archival_storage.search(...)`, etc.
- **Llamadas a cualquier otra tool del agente**: `web_search`, `calculator`, etc.
- **Salidas de todas esas funciones**: resultados de búsqueda, confirmaciones de éxito, errores.
- **Mensajes de sistema**: alertas de Memory Pressure, eventos de login del usuario, notificaciones de uploads, etc.
- **Eventos temporizados**: "son las 9:00 del 7 de febrero", para anclar al agente en el tiempo.

Por eso un solo turno conversacional puede generar varias entradas en la FIFO (llamada + resultado + respuesta + más function chaining).

### Umbrales de la FIFO (configurables)

El paper define dos umbrales que disparan acciones, con valores **a modo de ejemplo (configurables, no fijos)**:

| Umbral | Valor por defecto | Qué dispara |
|---|---|---|
| **Warning threshold** | ~70% del contexto | Inserta una *Memory Pressure Alert* en la FIFO para que el LLM consolide en Working Context lo que quiera preservar. |
| **Flush threshold** | ~100% del contexto | Dispara la expulsión: se evicta un bloque de mensajes (~50% del contexto por defecto) y se regenera el resumen recursivo. |

Cita textual (sección 2.2):

> *"e.g. 70% of the context window [...] e.g. 100% of the context window [...] e.g. 50% of the context window"*
>
> *(Traducción: "p. ej. el 70% de la ventana de contexto [...] p. ej. el 100% de la ventana de contexto [...] p. ej. el 50% de la ventana de contexto")*

Son ejemplos, no constantes hardcodeadas - cualquier implementación puede ajustarlos.

### El resumen recursivo (slot 0) y cómo se genera

Cuando se supera el flush threshold, ocurre lo siguiente:

1. El Queue Manager identifica los mensajes a expulsar (los más antiguos, ~50% del contexto por defecto).
2. **Se hace una llamada al LLM SEPARADA del ciclo principal del agente**, dedicada exclusivamente a regenerar el resumen.
3. El nuevo resumen ocupa el slot 0 de la FIFO.
4. Los mensajes expulsados desaparecen de la FIFO pero permanecen en Recall Storage (evocación).

Esta llamada de summarización es independiente porque **no cabría dentro del ciclo principal**: si el contexto del agente está al 100%, no se puede meter encima la instrucción "haz un resumen" más todo el material a resumir. Por eso el summarizer recibe su propio prompt mínimo, sin las System Instructions del agente principal.

Diseño propuesto para la llamada al summarizer (más rico que el del paper, evita duplicar info en Working Context):

```
[Llamada al summarizer - separada del agente principal]

System prompt:
  "Eres un resumidor. Recibes:
    - El Working Context actual del agente (lo que ya recuerda sin esfuerzo).
    - El resumen recursivo previo de la FIFO.
    - Los mensajes más antiguos de la FIFO que se van a expulsar.
   Devuelve:
    - Qué añadir al Working Context (si hay información nueva
      que merezca persistir y no esté ya).
    - El nuevo resumen recursivo, incorporando los mensajes
      expulsados al resumen previo."

Input:
  - Working Context: [contenido actual]
  - Resumen recursivo previo: [slot 0 de la FIFO]
  - Mensajes a expulsar: [los ~50% más antiguos]

Output:
  - working_context_updates: [...] (puede estar vacío)
  - new_recursive_summary: "..."
```

Pasarle el Working Context al summarizer es importante: así el resumidor **no propone añadir cosas que ya están guardadas**, y puede sugerir promociones de la FIFO al Working Context para datos relevantes a largo plazo.

### Ejemplo de FIFO en mitad de una conversación

```
FIFO Queue [
  (system) Resumen recursivo: "El usuario se llama Ana, vive en Madrid, le gusta el surf...";
  (user) "¿Te acuerdas dónde fuimos a comer?";
  (assistant) function_call: recall_storage.search(query="comida pacifica");
  (function_result) "[03/12/2024] 'Hay un Taco Bell al lado de la playa'";
  (assistant) function_call: send_message("¡Sí! Fuimos al Taco Bell de Pacifica.");
  (function_result) {"status": "ok"};
  (user) "Verdad! Por cierto, mañana voy con James";
  (system) Memory Pressure Warning: context at 75%;
  (assistant) function_call: working_context.append("Tiene un amigo/pareja llamado James");
  (function_result) {"status": "ok"};
  (assistant) function_call: send_message("¡Qué bien! ¿Os apetece repetir Taco Bell?");
  (function_result) {"status": "ok"};
]
```

Si Working Context es **lo que el agente recuerda sin esfuerzo** y Recall/Archival es **lo que tiene que ir a buscar**, la FIFO Queue es **el flujo en vivo de lo que está pasando ahora mismo**.

### Eventos automáticos (cómo "se acuerda" el agente solo)

El paper menciona en la sección 2.4 que MemGPT puede dispararse por **eventos temporizados que se ejecutan en un horario programado**, pero **no describe la implementación concreta**. Es otro detalle delegado en quien construya el sistema.

Cita textual (sección 2.4):

> *"timed events that are run on a regular schedule (allowing MemGPT to run 'unprompted' without user intervention)"*
>
> *(Traducción: "eventos temporizados que se ejecutan en un horario regular (permitiendo a MemGPT funcionar 'sin que se le pida' y sin intervención del usuario)")*

Existen **dos modalidades distintas** de eventos automáticos, que hay que distinguir bien:

| Modalidad | Trigger | Ejemplo | Útil para |
|---|---|---|---|
| **Eventos por reloj real (wall-clock)** | Una hora concreta del mundo real | "A las 9:00 AM, da los buenos días al usuario" | Saludos diarios, recordatorios programados, recapitulaciones semanales |
| **Eventos por número de iteraciones** | Cada N pasos del agente principal | "Cada 10 turnos, ejecuta una rutina de mantenimiento de memoria" | Consolidación periódica, revisión de calidad de respuestas, mantenimiento sin depender del reloj |

#### Modalidad 1: eventos temporizados (wall-clock)

Es la del paper: el evento se dispara cuando el reloj marca una hora concreta. Implementación con scheduler externo (cron, APScheduler, cola con ejecución diferida…).

#### Opciones de implementación típicas

| Mecanismo | Cuándo elegirlo |
|---|---|
| **Cron del SO / `systemd.timer`** | Prototipo simple, pocos agentes, granularidad ≥1 min. |
| **Scheduler de aplicación** (APScheduler, Celery Beat) | Demo o producción pequeña. Requiere proceso vivo. |
| **Cola con ejecución diferida** (Redis/BullMQ, RabbitMQ, SQS delayed) | Producción seria: persistente, escalable, sobrevive a caídas. |
| **Polling activo sobre BD** | Trivial de implementar pero desperdicia recursos. Solo para prototipos. |

Para producción real, **una cola persistente con ejecución diferida** es lo más sólido: cada agente registra sus eventos en una BD, un worker los consume cuando vencen, y los eventos sobreviven a reinicios del servicio.

Para un **prototipo o demo**, lo más eficiente es un **scheduler de aplicación** tipo APScheduler: se monta en pocas líneas, se gestiona desde el propio código Python sin servicios externos, ofrece granularidad de segundos y soporta tanto disparos puntuales como recurrentes (cron, intervalos, etc.). Suficiente para iterar rápido sin la complejidad operacional de Redis/RabbitMQ.

#### Cómo se conecta el evento al agente

Independientemente del mecanismo del timer, el flujo cuando "vence" el evento es siempre el mismo:

1. El scheduler dispara una función externa al LLM.
2. Esa función construye un **mensaje de sistema** del tipo:
   ```
   {
     "role": "system",
     "content": "Scheduled event: daily_greeting. Current time: 2024-02-07 09:00:00."
   }
   ```
3. El Queue Manager lo añade a la FIFO del agente.
4. Se lanza una inferencia del LLM con la ventana de contexto ya actualizada.
5. El LLM, al ver el mensaje de sistema, responde llamando a `send_message("¡Buenos días!")` (u otra función según el evento).

Lo elegante: **el LLM no necesita saber cómo se programó el evento**. Para él solo es un mensaje de sistema más que llega a la FIFO. Toda la fontanería del scheduler queda fuera de su contexto.

#### Modalidad 2: eventos por iteraciones (sleep-time agents de Letta)

En lugar de medir tiempo real, se cuentan **pasos del agente principal**: cada N inferencias o cada N turnos del usuario, se dispara un agente auxiliar (un *sleep-time agent*).

```
Agente principal → step 1 → step 2 → step 3 → ... → step N
                                                     │
                                                     ▼
                                          Sleep-time agent se activa
                                          (revisa memoria, consolida,
                                           sugiere acciones, etc.)
```

Casos de uso típicos:

- **Consolidación periódica**: "Cada 10 turnos, despierta a un agente que revisa la conversación y consolida lo aprendido en `core memory`."
- **Mantenimiento de calidad**: "Tras 20 mensajes sin actividad de búsqueda, despierta a un agente que sugiera buscar en archival si parece que falta contexto."
- **Auto-evaluación**: "Cada 50 turnos, despierta un agente que revise si la assistant del agente principal sigue siendo coherente con el comportamiento observado."

#### Cuándo elegir cada modalidad

| Necesitas... | Modalidad |
|---|---|
| Disparar algo a una hora real concreta (ej. 9:00 AM) | **Wall-clock** |
| Hacer mantenimiento que dependa del uso del agente, no del tiempo | **Por iteraciones** |
| Recordatorios al usuario | **Wall-clock** |
| Consolidar memoria tras intensa actividad conversacional | **Por iteraciones** |
| Ejecución sin dependencia de un proceso vivo | **Wall-clock** (cron del SO) |

Las dos modalidades **no son excluyentes** - un sistema serio puede usar ambas a la vez según la naturaleza de cada evento.

### `request_heartbeat`: cómo encadena el LLM múltiples acciones

Es un **argumento booleano especial** que el LLM puede incluir en cualquiera de sus llamadas a funciones. No afecta a la lógica de la función - es una instrucción **para MemGPT**, no para la función:

```
working_context.append(
  content="Birthday is February 7",
  request_heartbeat=true
)
```

Cita textual (sección 2.3):

> *"The LLM can request immediate follow-up LLM inference to chain function calls together by generating a special keyword argument (`request_heartbeat=true`) in its output."*
>
> *(Traducción: "El LLM puede solicitar una inferencia inmediata de seguimiento para encadenar llamadas a funciones generando un argumento especial (`request_heartbeat=true`) en su salida.")*

#### Por qué se llama "heartbeat"

En sistemas distribuidos, un *heartbeat* es un latido: una señal que mantiene un proceso activo. Aquí, el LLM está **pidiendo otro latido** - otra inferencia inmediata - en lugar de dormirse esperando un evento externo.

#### Las dos modalidades

| Caso | Qué hace MemGPT tras ejecutar la función |
|---|---|
| `request_heartbeat=true` | Añade el resultado al main context **e inmediatamente** lanza otra inferencia del LLM. El agente sigue trabajando. |
| Sin el flag (un *yield*) | Añade el resultado al main context y **se queda quieto** hasta el siguiente evento externo (mensaje del usuario, evento temporizado…). |

Cita textual (sección 2.4):

> *"If this flag is present, MemGPT will add the function output to main context and [continue] (as opposed to pausing processor execution). If this flag is not present (a yield), MemGPT will not run the LLM processor until the next external event trigger."*
>
> *(Traducción: "Si este flag está presente, MemGPT añadirá la salida de la función al main context y continuará (en lugar de pausar la ejecución del procesador). Si el flag no está presente (un yield), MemGPT no ejecutará el procesador LLM hasta el siguiente disparador de evento externo.")*

#### Cuándo usa el LLM cada modalidad

- **Con heartbeat**: cuando aún no ha terminado el turno y necesita el resultado de la función para decidir el siguiente paso. Ejemplos: paginar resultados, encadenar varias búsquedas (Figura 8: KV anidado), guardar algo en working context y seguir conversando.
- **Sin heartbeat**: cuando la llamada a función **es la acción final** del turno. Ejemplo típico: `send_message("Hola")` al usuario - después de eso no hay nada más que hacer hasta que el usuario conteste.

#### Ejemplo concreto: Figura 6 del paper

Usuario: "¿Quién ganó el primer Nobel de física?"

```
Inferencia 1:
  archival_storage.search("nobel physics", request_heartbeat=true)
  ↓ heartbeat → sigue

Inferencia 2 (ve la página 1, no encuentra la respuesta):
  archival_storage.search("nobel physics", page=2, request_heartbeat=true)
  ↓ heartbeat → sigue

Inferencia 3 (ya tiene la respuesta):
  send_message("Wilhelm Conrad Röntgen")
  ↓ yield → espera al usuario
```

Tres inferencias encadenadas en un solo turno gracias a `request_heartbeat=true`.

#### Por qué es crítico

Sin `request_heartbeat`, MemGPT solo podría hacer **una function call por turno del usuario**. No podría paginar, ni encadenar búsquedas, ni consolidar varias cosas en working context tras un *Memory Pressure*, ni razonar antes de hablar. Es lo que convierte a MemGPT de "agente reactivo de un paso" en "agente capaz de razonar en múltiples pasos antes de responder".

#### Riesgo: bucles infinitos

Como el LLM decide cuándo ceder, podría quedarse pidiendo heartbeats indefinidamente (p. ej. paginando para siempre). El paper no detalla protecciones, pero implementaciones reales suelen añadir:

- **Límite máximo de heartbeats encadenados por turno** (p. ej. 10).
- **Timeout** total por turno.
- **Detección de loops** (mismas llamadas repetidas sin progreso).

Otro detalle de implementación que el paper deja abierto.

#### `request_heartbeat` y `send_message` como mecanismos opcionales

Cuando se escribió el paper (2023), los LLMs no tenían **function calling robusto** ni **reasoning encadenado nativo**. Por eso MemGPT necesitaba dos trucos:

1. **`request_heartbeat`** para forzar que el LLM siguiera razonando tras una function call.
2. **`send_message(...)`** para que cualquier mensaje al usuario fuera una function call (no había forma fiable de mezclar texto y tool calls en un mismo output).

Con LLMs modernos (Claude Sonnet 4.6/Haiku 4.5, GPT-4o, Gemini, etc.) **ambos mecanismos son innecesarios por defecto**:

- Los modelos actuales hacen multi-step reasoning de forma nativa: hacen tool use, leen el resultado, hacen otra tool use, todo dentro de un mismo turno sin necesidad de un flag.
- También pueden emitir texto al usuario directamente, sin envolverlo en una función.

Por tanto, una implementación moderna debería tratar **ambos como opcionales / configurables**:

| Configuración | Cuándo usarla |
|---|---|
| **`request_heartbeat` desactivado (modo nativo)** | LLMs modernos con reasoning multi-paso fiable. Es el modo por defecto de Letta v1. |
| **`request_heartbeat` activado** | LLMs antiguos o más limitados (modelos pequeños open-source, modelos sin function calling robusto) que necesitan el flag explícito para encadenar acciones. |
| **`send_message` desactivado (texto directo al usuario)** | LLMs modernos que pueden alternar tool calls y texto en un mismo output. |
| **`send_message` activado** | LLMs que requieren que toda salida estructurada sea una function call. También útil para forzar un punto de control claro en logs/auditoría. |

En resumen: la arquitectura clásica del paper **sigue siendo válida y útil** - sobre todo si quieres correr MemGPT con LLMs pequeños o locales - pero **deja de ser necesaria** con los modelos punteros actuales. Letta refleja esto manteniendo ambos modos en paralelo.

### Qué es el Recall Storage

El **Recall Storage** ("almacenamiento de evocación") es el **historial completo de la conversación**. Vive **fuera de la ventana de contexto** y guarda **todos los mensajes** que han pasado por la FIFO Queue, también los que ya fueron eviccionados (expulsados). Es el "log" de la sesión: lo que se ha dicho, palabra por palabra.

#### Qué guarda

Cada mensaje (`HumanMessage`, `AIMessage`, `ToolMessage`, etc.) que entra en la FIFO se copia automáticamente a Recall en el mismo momento. Cuando luego un mensaje se evicta de la FIFO para liberar espacio, **no se pierde**: sigue accesible en Recall, solo deja de estar en el prompt.

#### Quién escribe

**El Queue Manager**, automáticamente. **El LLM no llama a ninguna función para escribir en Recall** — pasa solo. Esta es la diferencia más importante respecto a Archival: Recall es un *log pasivo* del sistema, no una decisión del agente.

#### Cómo lo usa el agente

Una única función, expuesta como tool:

```
recall_memory_search(query="...", page=N)   # buscar en el historial
```

Funciona análogamente a Archival (vector search + paginación), pero **solo sobre mensajes**, no sobre texto curado. La query típica es "¿qué dijo el usuario sobre X?" o "¿hablamos alguna vez de Y?".

#### Propiedades clave

| Propiedad | Explicación |
|---|---|
| **Tamaño ilimitado** | Toda la historia conversacional, sin límite. |
| **Escritura automática** | El LLM no decide qué entra: entra todo lo que pase por la FIFO. |
| **Solo lectura para el LLM** | El agente puede *buscar* pero no insertar ni borrar. |
| **Búsqueda semántica + paginación** | Misma mecánica que Archival: encoder + similaridad + páginas bajo demanda. |
| **Persistente entre sesiones** | Al reabrir el agente días después, todo el historial sigue siendo buscable. |

#### Recall vs Archival: dos cosas distintas

Conviene no confundirlos, porque ambos viven "fuera de la ventana" y ambos se buscan con vector search:

| | Recall Storage (evocación) | Archival Storage (documental) |
|---|---|---|
| **Qué guarda** | Toda la conversación que ha pasado por la FIFO | Solo lo que el LLM decide insertar explícitamente |
| **Quién escribe** | El Queue Manager, automáticamente | El LLM, vía `archival_memory_insert` |
| **Granularidad** | Mensajes individuales | Texto arbitrario (resúmenes, hechos, documentos) |
| **Para qué sirve** | Recuperar lo dicho ("¿qué dijo el usuario hace 3 sesiones?") | Conocimiento curado del agente o documentos externos |
| **Modificable por el LLM** | No, solo búsqueda | Sí, también escritura |

Regla rápida: **Recall = "transcripción"**, **Archival = "cuaderno de notas"**.

### Qué es el Archival Storage

El **Archival Storage** ("almacenamiento documental") es la memoria de largo plazo del agente: vive **fuera de la ventana de contexto**, es **ilimitada en tamaño** y el LLM accede a ella **solo bajo demanda** mediante funciones de búsqueda. Es el análogo al "disco" del SO: lenta de acceder, pero cabe todo.

#### Qué guarda

Texto arbitrario que el propio agente decide persistir: resúmenes que él mismo redacta, hechos sueltos que quiere recordar a largo plazo, documentos largos que el usuario le aporta, conclusiones que extrae de una conversación. **No** se guarda automáticamente cada mensaje (eso lo hace Recall Storage, ver más abajo) — Archival es escritura **explícita**: solo entra lo que el LLM llama `archival_memory_insert(...)`.

#### Cómo lo usa el agente

Dos funciones, ambas invocadas por el LLM como tools:

```
archival_memory_insert(content="...")             # escribir
archival_memory_search(query="...", page=N)       # leer (vector search, paginado)
```

La búsqueda es **semántica**: el storage indexa cada entrada con un encoder (Contriever en el paper; en implementaciones modernas, `text-embedding-3-small`, sentence-transformers, etc.) y `search` devuelve los top-K más similares al `query`. Si la página 1 no convence, el LLM pagina con `page=2` y reformula si hace falta — es **retrieval iterativo y auto-dirigido**, no un único disparo.

#### Propiedades clave

| Propiedad | Explicación |
|---|---|
| **Tamaño ilimitado** | A diferencia del Working Context, no hay budget de tokens — crece todo lo que haga falta. |
| **Read/Write vía funciones** | El LLM controla qué entra (`insert`) y qué sale (`search`). Nada se inyecta automáticamente. |
| **Búsqueda semántica** | Embeddings + similaridad coseno. No es `grep`. Si lo implementas con `substring match` deja de ser MemGPT. |
| **Paginación** | Los resultados se sirven en páginas; el agente decide cuándo seguir pidiendo. Esto es lo que evita el "lost in the middle". |
| **Persistente entre sesiones** | Se serializa a una BD vectorial (pgvector, Chroma, FAISS…). Sobrevive reinicios. |

#### Archival vs RAG clásico

Archival usa **RAG por debajo** (embeddings + similaridad), pero se diferencia en que el LLM **decide** cuándo buscar, **pagina** si los resultados no bastan y puede **escribir** en el corpus. En RAG clásico, el pipeline recupera siempre, mete los top-K en el prompt y el corpus es estático. (Sección 2 amplía esta comparación.)

### MemFS: memoria como filesystem versionado

**MemFS** es una extensión añadida en Letta (no está en el paper) que reorganiza la memoria del agente como un **sistema de ficheros versionado con git**. En lugar de tener bloques planos de texto, el agente puede crear "ficheros" y "carpetas" con jerarquía y mantener un historial completo de cambios.

#### Cómo cambia el modelo mental

| Modelo | Estructura | Edición | Historial |
|---|---|---|---|
| **Working Context clásico (paper)** | Un bloque plano de texto | `append` / `replace` sobre el bloque entero | Sin historial: cada cambio sobrescribe el anterior |
| **Core memory por bloques (Letta)** | Varios bloques etiquetados (`assistant`, `human`, …) | Operaciones por bloque | Sin historial: cada bloque guarda solo el estado actual |
| **MemFS** | Árbol de ficheros y carpetas | Operaciones tipo filesystem (crear, leer, escribir, mover, borrar) | **Versionado tipo git**: cada cambio queda registrado y se puede consultar |

#### Cómo lo usa el agente

El agente trata MemFS como cualquier otro conjunto de tools - con funciones tipo:

```
memfs.create(path="/projects/web/notes.md", content="...")
memfs.read(path="/projects/web/notes.md")
memfs.write(path="/projects/web/notes.md", content="...")
memfs.list(path="/projects/web/")
memfs.move(src="...", dst="...")
memfs.history(path="...")   ← consultar versiones anteriores
```

#### Para qué es útil

- **Proyectos largos**: organizar el conocimiento del agente en una jerarquía rica (`/users/ana/`, `/projects/portfolio/`, `/learning/spanish/`).
- **Trazabilidad**: poder ver **cómo evolucionó** una pieza de información ("¿qué decía el bloque del usuario hace 3 semanas?").
- **Rollback**: si el agente sobrescribió algo importante por error, se puede recuperar la versión anterior.
- **Compartir contexto entre agentes**: en sistemas multi-agente, varios agentes pueden leer/escribir sobre el mismo MemFS, usándolo como pizarra compartida.

### Working Context vs Archival Storage vs MemFS

Los tres son **memoria persistente del agente**, pero optimizan ejes distintos. **No son alternativas: coexisten**, y el agente decide caso a caso dónde guarda cada cosa nueva.

| Dimensión | Working Context (Core Memory) | Archival Storage | MemFS |
|---|---|---|---|
| **Dónde vive** | Dentro de la ventana de contexto | Fuera (BD vectorial) | Fuera (filesystem versionado) |
| **Tamaño** | Pequeño (budget de tokens) | Ilimitado | Medio-grande |
| **En contexto siempre** | ✅ Sí, visible en cada inferencia | ❌ No (se busca) | ❌ No (se navega) |
| **Cómo se accede** | Lectura directa (está en el prompt) | `archival_memory_search` (vector search) | `memfs.read` / `memfs.list` (por path) |
| **Cómo se escribe** | `core_memory_append/replace` por bloque | `archival_memory_insert` | `memfs.create/write/move` |
| **Estructura** | Bloques etiquetados planos (`assistant`, `human`, …) | Entradas sueltas indexadas por embedding | Árbol jerárquico de ficheros y carpetas |
| **Búsqueda** | No hace falta (se ve directo) | Semántica (similaridad) | Estructural (path) + por contenido |
| **Versionado** | ❌ No | ❌ No | ✅ Sí, tipo git |
| **Coste por uso** | Tokens del contexto en cada turno | Una tool call (round-trip al LLM) | Una tool call (round-trip al LLM) |

#### Regla mental para decidir dónde va cada cosa

| Si el dato es… | Va a… |
|---|---|
| Crítico, debe verse en **cada** turno (identidad, preferencias clave, estado actual del usuario) | **Working Context** |
| Voluminoso, no sabes a priori cuándo lo necesitarás, búsqueda por significado ("¿qué hablamos de X hace meses?") | **Archival Storage** |
| Estructurado, jerárquico, evoluciona en el tiempo y quieres ver su historial (notas de proyecto, especificaciones, bitácoras) | **MemFS** |

#### Ejemplo concreto

Un agente que ayuda al usuario durante meses con varios proyectos:

- **Working Context**: `"El usuario es Máximo, prefiere español, trabaja con Astro + LangGraph"` → siempre visible.
- **Archival Storage**: cada conversación importante resumida y persistida (`archival_memory_insert("El usuario decidió usar pgvector porque ...")`); luego `archival_memory_search("decisión sobre vector DB")` la recupera meses después.
- **MemFS**:
  - `/projects/portafolio/decisiones.md` → bitácora versionada de decisiones de arquitectura
  - `/users/maximo/objetivos-q2.md` → objetivos del trimestre
  - `memfs.history("/projects/portafolio/decisiones.md")` permite ver cómo evolucionaron las decisiones.

Los tres trabajan en conjunto: identidad inmediata en Working Context, conocimiento buscable por contenido en Archival, conocimiento organizado y versionado en MemFS.

---

## 2. La jerarquía de memoria (Figura 3)

### Main context (lo que el LLM "ve" - análogo a la RAM)

![Figura 3. En MemGPT, un procesador LLM de contexto fijo se aumenta con un sistema de memoria jerárquica y funciones que le permiten gestionar su propia memoria.](./MemGPT-Figure3.png)

Son los *prompt tokens* dentro de la ventana de contexto, divididos en tres bloques:

| Bloque | Permiso | Función |
|---|---|---|
| **System Instructions** | Read-only | Prompt fijo: cómo usar la memoria, qué funciones existen |
| **Working Context** | Read/Write vía funciones | Bloque de texto libre con datos clave (p. ej. "novio se llama James"). Es la "libreta" del agente |
| **FIFO Queue** | Read/Write vía Queue Manager | Historial rodante de mensajes recientes + un resumen recursivo en la primera posición |

### External context (fuera de la ventana - análogo al disco)

| Almacenamiento | Qué guarda | Cómo se escribe / lee |
|---|---|---|
| **Recall Storage (evocación)** | Toda la base de datos de mensajes (todo lo que ha pasado por la conversación) | Escrito automáticamente por el Queue Manager; leído por el LLM con funciones de búsqueda |
| **Archival Storage (documental)** | Texto arbitrario de cualquier longitud (documentos, notas) con búsqueda vectorial | Read/write totalmente vía funciones del LLM |

### ¿Archival Storage es lo mismo que RAG?

Pregunta legítima al ver "embeddings + búsqueda por similaridad sobre documentos". La respuesta corta: **sí, Archival Storage usa RAG por debajo, pero el agente lo opera distinto**.

**RAG clásico** ("chatear con tus PDFs"):

1. Trocear documentos en chunks.
2. Embeber cada chunk → vector.
3. Indexar en una BD vectorial (Chroma, FAISS, pgvector…).
4. En cada query del usuario: embeber la pregunta → buscar top-K chunks → **inyectarlos automáticamente en el prompt** → el LLM genera respuesta.

En RAG clásico, **el pipeline decide buscar**: el usuario pregunta, el sistema recupera, el LLM solo lee y responde. La recuperación es **opaca y de un solo paso** - lo que entra al prompt es lo que entra, no hay vuelta atrás.

**MemGPT Archival** hace exactamente lo mismo en el storage layer (vector search sobre embeddings), pero la búsqueda está expuesta como **tool** (`archival_memory_search(query, page)`) que el **LLM decide** invocar. Eso cambia tres cosas:

1. **El agente decide si necesita buscar.** Si la pregunta es trivial, no llama a la tool y se ahorra el round-trip. RAG clásico siempre recupera, aunque sobre.
2. **El agente puede paginar.** Si la primera página de resultados no convence, llama a la tool de nuevo con `page=2`, o reformula la query. Esto es lo que valida la Figura 5: con K alto (muchos docs distractores), MemGPT mantiene accuracy plana porque pagina; el RAG clásico cae porque mete los top-K en el prompt y el LLM se pierde "in the middle".
3. **El agente puede escribir en Archival.** `archival_memory_insert` deja al LLM persistir conocimiento que generó él mismo (resúmenes, conclusiones, lo que el usuario le contó hace 3 sesiones). En RAG clásico el corpus es estático.

En resumen: **Archival = RAG (motor de retrieval) + control del LLM (cuándo, cómo, cuántas veces) + escritura por el agente**. La diferencia no está en cómo se almacena, está en quién decide cuándo recuperar y qué hacer si lo recuperado no basta.

> **Implicación práctica:** si implementas Archival Storage con `query in content` (substring) en lugar de embeddings + cosine, **no estás implementando MemGPT** - estás implementando un grep con esteroides. El paper asume retrieval semántico (Contriever en su caso); cualquier réplica honesta necesita un encoder real (`text-embedding-3-small`, sentence-transformers, etc.).

---

## 3. El ciclo de funcionamiento

### Cuando llega un evento (mensaje del usuario, alerta del sistema, evento programado…):

1. **Queue Manager** añade el mensaje a la FIFO Queue y lo guarda en Recall Storage (evocación).
2. Concatena los tres bloques del main context y dispara la inferencia del LLM.
3. La salida del LLM **no se interpreta como respuesta directa al usuario** - se interpreta como una **llamada a función** por el Function Executor.
4. La función se ejecuta (puede ser: enviar un mensaje al usuario, escribir en working context, buscar en archival, etc.) y su resultado se devuelve al main context.
5. Si la función llevaba el flag `request_heartbeat=true`, el LLM se vuelve a invocar inmediatamente (**function chaining**, multi-paso). Si no, MemGPT espera al siguiente evento.

### Gestión de desbordamiento de contexto

- Cuando los tokens superan el **warning threshold** (~70%), se inserta una *Memory Pressure Alert* (Figura 1). El LLM, viendo la alerta, decide qué guardar de la FIFO en working context o archival (documental) antes de perderlo.
- Cuando se supera el **flush threshold** (100%), el Queue Manager evicta (expulsa) mensajes antiguos, genera un nuevo resumen recursivo y libera espacio. Lo eviccionado (expulsado) **no se pierde**: sigue en Recall Storage (evocación) y se puede recuperar con búsquedas paginadas.

---

## 4. Cómo se ve en la práctica (las figuras)

- **Figura 1**: el usuario menciona "mi novio James me hizo una tarta el 7 de febrero". Llega *Memory Pressure*. El LLM llama dos veces a `working_context.append(...)` para guardar "Birthday is February 7" y "Boyfriend named James" antes de que se eviccionen (expulsen).
- **Figura 2**: días después el usuario dice "fuimos a Six Flags". Como esos mensajes ya no están en contexto, el LLM llama `recall_storage.search("six flags")` y trae 3 resultados pasados; con eso responde con personalización ("¿fuiste con James?").
- **Figura 4**: el usuario dice "James y yo rompimos". El LLM llama `working_context.replace("Boyfriend named James", "Ex-boyfriend named James")` - **edita** su propia memoria persistente.
- **Figura 6**: para responder "¿quién ganó el primer Nobel de física?", el LLM hace `archival_storage.search("nobel physics")`, recibe 10 de 124 resultados (página 1/13), pagina con `page=2`, encuentra la respuesta y contesta "Wilhelm Conrad Röntgen".
- **Figura 8**: tarea de KV anidado - el LLM encadena 3 búsquedas (`831...ea5` → `5b8...4c3` → `f37...617`) hasta que el resultado ya no es una clave. Esto es **multi-hop retrieval** auto-dirigido.

---

## 5. Por qué funciona (lo que dicen las figuras 5 y 7)

- **Document QA (Fig. 5)**: los baselines (GPT-4, GPT-4 Turbo) caen en precisión cuando metes muchos documentos al contexto (lost-in-the-middle). MemGPT mantiene precisión constante porque no apila documentos - los pagina bajo demanda.
- **Nested KV (Fig. 7)**: GPT-4 baseline cae a 0% en 3 niveles de anidamiento. MemGPT con GPT-4 mantiene **100% en todos los niveles** porque puede iterar tantas búsquedas como necesite.

---

## 6. Resumen en una frase

**MemGPT convierte al LLM en un agente que se auto-administra la memoria mediante function calls, con una jerarquía de dos niveles (contexto = RAM, archival/recall (documental/evocación) = disco) y un control de flujo dirigido por eventos, logrando que un modelo de 8k tokens se comporte como si tuviera contexto ilimitado.**

---

## 7. TODO / Limitaciones identificadas

### Eliminación selectiva en la FIFO

El paper **no contempla** ningún mecanismo para que el LLM elimine entradas concretas de la FIFO. La única forma de que un dato salga de la FIFO es la **expulsión automática por antigüedad** (FIFO pura), en bloque, cuando se supera el flush threshold.

Esto deja sin resolver dos casos donde sería útil borrar selectivamente:

1. **Tangentes irrelevantes**: el usuario dice "espera, me llaman por teléfono" y al rato vuelve. Esos mensajes ocupan FIFO hasta que les toque por antigüedad - el LLM no puede marcarlos como ruido y eliminarlos.
2. **Salidas verbosas de tools intermedias**: si el agente hace varios `grep`/`ls`/`tree` para encontrar un path, todas esas salidas (a menudo cientos de líneas) quedan en la FIFO hasta su expulsión natural, aunque solo el path final sea relevante.

La compensación parcial que ofrece MemGPT:
- Consolidar lo importante en Working Context antes de la expulsión.
- Confiar en que el resumen recursivo, al regenerarse, omitirá lo irrelevante.

Pero no equivale a una eliminación selectiva real. Sistemas posteriores (Letta, MemoryOS, etc.) han explorado FIFOs con prioridades o edición selectiva - área abierta de investigación.

#### Momento ideal para una limpieza selectiva: cuando llega la Memory Pressure Alert

La llegada de una *Memory Pressure Alert* es el **momento perfecto** para que el LLM haga limpieza selectiva: ya tiene la atención puesta en gestionar memoria, está repasando la FIFO para decidir qué consolidar en Working Context, y le sobra contexto para razonar (estamos al ~70%, no al 100%).

Una extensión natural sería darle al LLM, junto con la alerta, una función adicional tipo `fifo.delete_messages(message_ids=[...])` para borrar selectivamente entradas que considere ruido. El flujo sería:

1. Llega la alerta al 70%.
2. El LLM consolida lo importante en Working Context (igual que ahora).
3. **Adicionalmente**, identifica entradas claramente irrelevantes (tangentes, salidas verbosas de tools intermedias) y las borra con `fifo.delete_messages(...)`.
4. Resultado: el contexto baja por debajo del 70% sin necesidad de esperar al flush automático, y el resumen recursivo nunca llega a contaminarse con ese ruido.

Esto convertiría a MemGPT en un sistema **proactivo** en lugar de reactivo: en vez de esperar al flush mecánico al 100%, el LLM podría mantener la FIFO "limpia" continuamente.

### Contador de búsquedas por página + límite por página

El paper **no contempla** ningún contador de llamadas en las funciones de búsqueda. Esto deja dos problemas sin resolver:

1. **Bucles infinitos**: el LLM podría repetir indefinidamente la misma búsqueda en `request_heartbeat=true`.
2. **Terminación prematura** (problema observado en el paper, sección 3.2.1): el LLM tiende a parar antes de agotar los resultados, descartando páginas que en realidad eran relevantes.

#### Propuesta: contador por página + límite por página

Cada vez que el LLM consulta una página concreta de una búsqueda (`archival_storage.search("X", page=N)`), MemGPT mantiene internamente un **contador asociado a esa página**. El contador se incrementa en cada lectura.

Caso de uso que esto habilita:

1. El LLM busca "nobel physics", lee la página 1 → contador página 1 = 1.
2. Cree que no está la respuesta, salta a la página 2 → contador página 2 = 1.
3. Sigue paginando hasta la 13 → todas con contador 1.
4. Tras revisar todo, concluye que la **página 1 era la más relevante**.
5. Vuelve a leer la página 1 → contador página 1 = 2.

Sobre cada contador habría un **límite máximo por página** (p. ej. 3). Si el LLM intenta leer una página más veces, MemGPT le devuelve un mensaje de sistema del tipo:

```
"You have already read page 1 the maximum number of times (3).
Either consolidate the relevant info into working context, or move on."
```

Esto previene bucles sin impedir el patrón legítimo de "vuelvo a una página tras descartarla", que es exactamente el caso del LLM que primero pasa de largo y luego reconsidera.

También se podría añadir un **límite global de llamadas a búsqueda por turno** (p. ej. 20 búsquedas máximo por evento de usuario), como red de seguridad adicional.

### Prompts intermedios para evitar terminación prematura

El paper observa explícitamente (sección 3.2.1) que MemGPT **se rinde antes de agotar los resultados**:

> *"we observe that MemGPT will often stop paging through retriever results before exhausting the retriever database"*
>
> *(Traducción: "observamos que MemGPT a menudo deja de paginar resultados antes de agotar la base de datos del retriever")*

La única mitigación que el paper aplica es a nivel de prompt inicial (apéndice 6.1.6 para la tarea KV: *"DO NOT STOP SEARCHING UNTIL..."*). Pero eso es una instrucción **fija al principio**.

#### Propuesta: prompts dinámicos de motivación

Cuando MemGPT detecta que el LLM está a punto de rendirse (p. ej. genera un `send_message` antes de haber paginado al menos N veces, o tras una sola búsqueda fallida), inyecta un mensaje de sistema **antes de la siguiente inferencia** del tipo:

```
"You have only checked 2 of 13 pages. The answer may be in a later page.
Consider continuing the search before responding to the user."
```

O, más sutil, cuando una búsqueda devuelve N resultados pero el LLM solo lee la primera página:

```
"There are 11 more pages of results available. Continue searching
if you haven't found the relevant information yet."
```

Esto convertiría la lucha contra la terminación prematura en algo **adaptativo**, en vez de depender solo del prompt inicial. Combinado con los contadores de página, formaría un sistema de control de búsqueda mucho más robusto que el del paper original.

### Desbordamientos masivos en una sola entrada (>100% de golpe)

El paper **asume crecimiento incremental** del contexto: un mensaje a la vez, una function output a la vez. El flush al 100% expulsa el ~50% del contexto, lo cual basta cuando el desbordamiento es pequeño.

**Caso no contemplado**: una entrada masiva en un solo evento. Por ejemplo:

- Estado: FIFO al 95%.
- Usuario pega el texto entero de un libro (50.000 tokens en una ventana de 8.000).
- Si MemGPT añade ese mensaje a la FIFO sin más, el contexto pasaría a un ~120% (o muchísimo más). El flush convencional expulsaría el 50% del contexto, pero el nuevo mensaje **sigue sin caber**.

El paper no aborda este caso.

#### Propuesta: medición previa + flush proactivo + chunking

Antes de añadir cualquier nuevo input a la FIFO (sea mensaje del usuario, llamada a función o salida de función), MemGPT debería:

1. **Medir el número de tokens** del input que va a añadir.
2. Calcular si `tokens_actuales + tokens_input > 100% de la ventana`.
3. Si sí, **disparar el flush proactivamente** antes de la inserción, expulsando suficiente contexto para que la nueva entrada quepa.
4. Si **incluso tras flushear todo lo que se pueda** la entrada sigue sin caber (caso del libro entero), se necesita una estrategia adicional:
   - **Chunking**: trocear la entrada y procesarla por partes, posiblemente vía `archival_storage.insert(...)` (volcando el libro al disco) y dejando solo un puntero o resumen en la FIFO.
   - **Rechazo controlado**: devolver al usuario un mensaje de sistema indicando que la entrada es demasiado grande y sugerir cargarla vía archival.
   - **Truncado con aviso**: cortar la entrada y avisar al usuario, peor opción.

Esto convertiría la gestión de memoria en **dos capas**:

- **Capa reactiva** (la del paper): warning al 70%, flush al 100% sobre crecimiento incremental.
- **Capa proactiva** (faltante en el paper): medición previa de cualquier input antes de añadirlo, con flush o chunking si haría desbordar.

---

## 8. Cómo replicar los experimentos

El paper publica todos los datasets y código necesarios para reproducir los benchmarks.

### Repositorios y datasets

- **Repositorio del paper**: https://research.memgpt.ai (código original, datasets, embeddings).
- **Letta** (sucesor productivo, mantenido por los autores): https://github.com/letta-ai/letta. Más actualizado y recomendado para implementaciones nuevas.
- **Datasets en Hugging Face**: https://huggingface.co/MemGPT/datasets - contiene el MSC aumentado, el Nested KV y los embeddings de Wikipedia precalculados.

> ⚠️ **Importante**: aunque Letta implementa la arquitectura MemGPT, **los benchmarks del paper no son ejecutables tal cual desde el repo de Letta**. El propio issue [#3115](https://github.com/letta-ai/letta/issues/3115) confirma que *"Letta carece de cualquier benchmark estandarizado o código de evaluación"* para memoria. Para replicar los experimentos hay que reconstruir el pipeline de evaluación a mano usando los datasets de Hugging Face.

### Datasets por experimento

#### Multi-Session Chat (MSC) - para experimentos conversacionales (DMR y Conversation Opener)

- **Origen**: Xu et al. (2021), *"Beyond goldfish memory: Long-term open-domain conversation"* (arXiv:2107.07567).
- **Estructura**: 5 sesiones por par de usuarios + ~12 mensajes por sesión, cada usuario interpretando una assistant consistente.
- **Versión aumentada por los autores**: añaden una **sesión 6** con un par pregunta-respuesta sobre algo de las sesiones 1-5 (generado vía LLM con el prompt del apéndice 6.1.3).
- **Métricas**:
  - DMR: ROUGE-L (recall) + LLM-as-a-judge (apéndice 6.1.2).
  - Conversation Opener: SIM-1, SIM-3, SIM-H sobre las personas gold.
- **Acceso**: el MSC original está en parlAI/Hugging Face; la versión aumentada con sesión 6 se publica en el repo del paper.

#### NaturalQuestions-Open - para Document QA

- **Origen**: tarea retriever-reader de Liu et al. (2023a), *"Lost in the middle"* (arXiv:2307.03172).
- **Estructura**: 50 preguntas muestreadas para evaluación, con respuestas extraíbles de Wikipedia (dump de finales de 2018).
- **Infraestructura**: PostgreSQL + `pgvector` con índice HNSW para búsqueda vectorial sub-segundo.
- **Embeddings**: `text-embedding-ada-002` de OpenAI sobre **20M artículos de Wikipedia** (los autores publican estos embeddings precalculados).
- **Métrica**: accuracy con LLM-as-a-judge (prompt del apéndice 6.1.5).
- **Recurso más pesado**: requiere montar la BD + cargar todos los embeddings.

#### Nested Key-Value Retrieval - propuesto en el paper

- **Origen**: extensión de la tarea KV sintética de Liu et al. (2023a).
- **Estructura**:
  - 140 pares clave-valor (cada uno es un UUID de 128 bits).
  - Total ~8.000 tokens (calibrado para llenar la ventana de GPT-4 base).
  - Niveles de anidamiento de 0 a 4 (el valor de una clave puede ser otra clave).
  - 30 configuraciones de orden distintas.
- **Métrica**: accuracy.
- **Acceso**: dataset publicado por los autores.
- **Es el más rápido de ejecutar**: ideal para iterar tu implementación.

##### Cómo se construye una configuración

Cada configuración contiene una **cadena guía** de 5 saltos plus un terminal:

```
k0 → k1 → k2 → k3 → k4 → terminal_no_clave
```

Es decir, 5 pares clave-valor donde el valor de cada uno es la clave del siguiente, excepto el último, cuyo valor (`terminal`) **no aparece como clave** en ningún par del dataset. Los otros 135 pares son distractores: claves frescas con valores aleatorios que no chocan con la cadena.

##### Cómo se generan las 5 queries por configuración

Cada nivel parte de un punto distinto de la misma cadena, exigiendo más o menos saltos hasta el terminal:

| Nivel | Start key | Cadena de búsquedas | Saltos |
|------:|-----------|---------------------|-------:|
| 0 | k4 | k4 → terminal | 1 |
| 1 | k3 | k3 → k4 → terminal | 2 |
| 2 | k2 | k2 → k3 → k4 → terminal | 3 |
| 3 | k1 | k1 → k2 → k3 → k4 → terminal | 4 |
| 4 | k0 | k0 → k1 → k2 → k3 → k4 → terminal | 5 |

Las 5 queries son **independientes** entre sí (cada una abre su propia conversación con el agente). Lo que comparten es la archival memory: los 140 pares idénticos. Por eso todas las queries OK de una misma configuración predicen el **mismo terminal** - es el final único de la cadena guía.

##### Cómo sabe el agente que ha terminado

Hay una sutileza: tras llegar al terminal, el agente hace **una búsqueda extra** del propio terminal en archival. Esa búsqueda devuelve un solo match (el par donde aparece como **valor**, no como clave) - eso confirma que no es clave de nadie y la cadena se acaba ahí. Por eso el contador de búsquedas es `nivel + 2` y no `nivel + 1`. Es la heurística que el system prompt del apéndice 6.1.6 dicta literalmente:

> "DO NOT STOP SEARCHING UNTIL YOU VERIFY THAT THE VALUE IS NOT A KEY."

Es lo que distingue MemGPT del baseline: con tools de archival el agente puede iterar la búsqueda; sin ellas, el LLM tiene que resolver toda la cadena leyendo el JSON del prompt y se atasca a partir del nivel 1-2.

### Modelos usados como baseline

| Nombre en el paper | Endpoint OpenAI | Ventana de contexto |
|---|---|---|
| GPT-4 | `gpt-4-0613` | 8.192 tokens |
| GPT-4 Turbo | `gpt-4-1106-preview` | 128.000 tokens |
| GPT-3.5 Turbo | `gpt-3.5-turbo-1106` | 16.385 tokens |

Estos modelos pueden estar deprecated hoy. Para replicar con modelos actuales (Claude Sonnet 4.6/Haiku 4.5, GPT-4o, etc.) las métricas son comparables siempre que la ventana de contexto sea similar.

### Prompts y personas (apéndice 6.1 del paper)

| Apéndice | Contenido |
|---|---|
| 6.1.1 | assistant MemGPT para tareas de chat (DMR) |
| 6.1.2 | Prompt del LLM-judge para DMR / Opener |
| 6.1.3 | Prompt para generar el dataset DMR (self-instruct) |
| 6.1.4 | assistant MemGPT para document analysis |
| 6.1.5 | Prompt del LLM-judge para document analysis |
| 6.1.6 | assistant MemGPT para K/V tasks |

### Plan mínimo de replicación

Orden recomendado de menor a mayor coste:

1. **Nested KV** - sintético, sin retrieval externo, rápido de ejecutar. Ideal para validar tu implementación de function chaining + paginación.
2. **DMR (MSC)** - solo necesitas las sesiones 1-5 + la pregunta de la sesión 6. Validas Working Context, Recall Storage y persistencia entre sesiones.
3. **Document QA** - el más pesado: PostgreSQL + pgvector + embeddings precalculados de 20M artículos. Validas archival storage a escala real.

Empezar por (1) y (2) te permite tener un MemGPT funcional sin montar infraestructura de retrieval, y dejar (3) para cuando ya tengas la lógica del agente sólida.

---

## 9. El repositorio de Letta (sucesor productivo de MemGPT)

**Letta** (https://github.com/letta-ai/letta) es la implementación oficial mantenida por los autores del paper. El repo histórico `cpacker/MemGPT` ahora **redirige** a este - son el mismo proyecto, simplemente renombrado. Es la referencia recomendada para implementaciones nuevas.

### Mapa de renombramientos paper → Letta

| Paper (arXiv 2310.08560) | Letta (actual) |
|---|---|
| Working Context | **Core Memory** (bloques etiquetados: `assistant`, `human`, custom) |
| Recall Storage | **Conversation history** / `conversation_search` |
| Archival Storage | **Archival Memory** (`archival_memory_insert`, `archival_memory_search`) |
| `working_context.append` | `core_memory_append(label, content)` |
| `recall_storage.search` | `conversation_search(query, roles, limit, start_date, end_date)` |
| FIFO Queue + warning/flush | `summarizer_settings.memory_warning_threshold` + `summarize_messages_inplace()` |
| `request_heartbeat` | **Mantiene el mismo nombre** (aunque opcional en v1) |

### Qué está implementado

✅ **Arquitectura completa del paper**:

- Jerarquía de memoria (Core Memory + Conversation history + Archival Memory).
- Function calling auto-dirigido.
- Queue Manager con summarización recursiva (`summarize_messages_inplace()` en `agent.py`).
- Warning threshold con `agent_alerted_about_memory_pressure` para evitar avisos duplicados.
- Persistencia con SQLAlchemy + PostgreSQL/pgvector.
- `request_heartbeat` para function chaining (en modo legacy).

✅ **Extensiones sobre el paper**:

- **Bloques etiquetados** en Core Memory (assistant/human/custom).
- **Sleep-time agents**: eventos por iteraciones, no solo por wall-clock.
- **MemFS**: memoria estructurada como filesystem versionado con git.
- **Multi-agente**: agentes pueden crear subagentes y comunicarse entre sí.
- **MCP tools**: integración con Model Context Protocol para usar servidores MCP como herramientas.
- **Letta v1**: rearquitectura del agent loop que deprecia `request_heartbeat` y `send_message` aprovechando reasoning nativo de los LLMs modernos.

### Qué NO está en el repo

❌ **Benchmarks ejecutables del paper**: el repo de Letta no incluye un pipeline de evaluación para reproducir los experimentos del paper. El issue [#3115](https://github.com/letta-ai/letta/issues/3115) lo confirma explícitamente: *"Letta carece de cualquier benchmark estandarizado o código de evaluación"* para memoria. Los datasets sí están en https://huggingface.co/MemGPT/datasets, pero el código de evaluación hay que reconstruirlo.

❌ **Eventos wall-clock reales**: el cron por hora del reloj **no está en el core open-source**. Vive en LettaBot (su producto SaaS de pago). Para tener eventos a horas concretas hay que añadir un scheduler externo (APScheduler, cron del SO, etc.).

### Letta v0 vs Letta v1: dos arquitecturas en paralelo

Letta mantiene **dos modos** convivientes:

| Modo | Comportamiento | Cuándo usarlo |
|---|---|---|
| **Letta v0 (clásico MemGPT)** | `request_heartbeat` y `send_message` activos como en el paper | LLMs antiguos / pequeños / locales que no tienen reasoning multi-paso fiable |
| **Letta v1 (`letta_v1_agent`)** | Reasoning nativo del LLM, `request_heartbeat` y `send_message` deprecados | LLMs modernos (Claude 4.x, GPT-4o, Gemini) con tool use y multi-step reasoning robustos |

### Archivos clave del repo

Si quieres bucear directamente en el código:

| Path | Contenido |
|---|---|
| `letta/agent.py` | Agent loop, heartbeat, summarización, memory pressure |
| `letta/functions/function_sets/base.py` | `core_memory_append/replace`, `archival_memory_insert/search`, `conversation_search` |
| `letta/functions/function_sets/multi_agent.py` | Funciones de multi-agente (envío de mensajes entre agentes) |
| `letta/schemas/memory.py` | Clases `Memory`, `BasicBlockMemory`, `ChatMemory` |
| `letta/schemas/passage.py` | Pasajes archival con embeddings |
| `letta/orm/` | Capa de persistencia (SQLAlchemy + PostgreSQL) |

### Documentación oficial relevante

- [Concepts: MemGPT](https://docs.letta.com/concepts/memgpt/) - explica el mapeo paper ↔ Letta.
- [Heartbeats](https://docs.letta.com/guides/agents/heartbeats) - cómo funciona el chaining hoy.
- [Sleep-time Agents](https://docs.letta.com/guides/agents/architectures/sleeptime/) - eventos por iteraciones.
- [Letta v1 Agent](https://www.letta.com/blog/letta-v1-agent) - post sobre la rearquitectura del agent loop.
- [MemGPT is now part of Letta](https://www.letta.com/blog/memgpt-and-letta) - anuncio del rebranding.

### Recomendación práctica

Para implementar MemGPT a día de hoy:

1. **Si quieres aprender la arquitectura** → empieza con Letta v0 (clásico): te obliga a entender heartbeat, send_message y el ciclo completo.
2. **Si quieres construir un agente real con LLMs modernos** → usa Letta v1: aprovecha el reasoning nativo y tendrás menos fricción.
3. **Si quieres replicar los benchmarks del paper** → tendrás que escribir tú el pipeline de evaluación sobre los datasets de Hugging Face - el repo no lo trae.

---

## 10. Implementar MemGPT desde cero: comparativa de frameworks

Si en lugar de usar Letta directamente quieres **construir tu propia implementación** de MemGPT en el ecosistema LangChain (por aprendizaje o para tener control total), las tres opciones evaluadas son **LangChain core**, **LangGraph** y **DeepAgents**. Resumen de la comparativa:

### Tabla por componente (asumiendo Graphiti como backend de almacenamiento)

Como Graphiti puede usarse encima de **cualquier** framework de agente, lo justo es comparar los tres asumiendo que **todos** lo integran como backend. Esto cambia el panorama en las filas de Recall/Archival/Persistencia, pero **no afecta al resto** (Queue Manager, FIFO, eventos, etc. siguen siendo responsabilidad del runtime).

> **Nota sobre la elección Graphiti vs mem0**: ambas librerías cubren la misma necesidad (Recall + Archival con búsqueda híbrida) y son **alternativas, no complementarias**. La elección aquí es **Graphiti** porque su modelo bi-temporal encaja con la naturaleza de MemGPT (datos que evolucionan en el tiempo, como el ejemplo "Boyfriend named James" → "Ex-boyfriend named James" de la Figura 4) y porque supera al propio MemGPT en el benchmark DMR (94.8% vs 93.4%). Si en otro proyecto interesa simplicidad de integración por encima del razonamiento temporal, mem0 sería el sustituto natural.

| Componente MemGPT | LangChain core + Graphiti + libs auxiliares | LangGraph + Graphiti + libs auxiliares | DeepAgents + Graphiti + libs auxiliares |
|---|---|---|---|
| **System Instructions** | ✅ `SystemMessage` | ✅ `SystemMessage` en `MessagesState` | ⚠️ Estático, sin separación read-only |
| **Core Memory (antiguo Working Context, bloques etiquetados)** | ✅ Pydantic custom (~50 líneas) | ✅ Pydantic custom + tools LangGraph (~50 líneas) | ⚠️ Pydantic custom pero peleando con `AGENTS.md` |
| **FIFO Queue + slot 0 (resumen recursivo)** | ❌ Memorias deprecated en v0.3.1 | ⚠️ `langmem.SummarizationNode` cubre 80% (bugs con tools: #118, #111, #126) | ⚠️ `SummarizationMiddleware` (un solo umbral 85%) |
| **Conversation Search (antiguo Recall Storage)** | ✅ via Graphiti (knowledge graph temporal + híbrida) | ✅ via Graphiti (knowledge graph temporal + híbrida) | ✅ via Graphiti (knowledge graph temporal + híbrida) |
| **Archival Memory (antiguo Archival Storage)** | ✅ via Graphiti (semántico + BM25 + graph traversal) | ✅ via Graphiti (semántico + BM25 + graph traversal) | ✅ via Graphiti (semántico + BM25 + graph traversal) |
| **Queue Manager (umbrales 70/100%)** | ⚠️ `litellm.token_counter` + lógica manual | ⚠️ `litellm.token_counter` + LangChain Middleware (`before_model`/`after_model`) | ⚠️ `litellm.token_counter` + middleware propio |
| **`request_heartbeat`** | ❌ Manual sobre `AgentExecutor` | ⚠️ `instructor` para tool tipado + edge condicional manual | ❌ No existe |
| **Eventos wall-clock** | ✅ APScheduler 3.x | ✅ APScheduler 3.x (sin pagar LangGraph Platform) | ✅ APScheduler 3.x |
| **Eventos por iteraciones (sleep-time agents)** | ⚠️ Contador manual | ✅ LangChain Middleware (~10 líneas) | ⚠️ Contador manual |
| **Persistencia entre sesiones** | ✅ Graphiti persiste memoria + falta serializar Core Memory | ✅ Checkpointer + Store + Graphiti | ✅ `CompositeBackend` + Graphiti |
| **Multi-agente** | ❌ | ✅ Subgrafos nativos | ⚠️ Subagentes efímeros, sin memoria compartida |
| **MCP** | ❌ | ✅ `langchain-mcp` | ✅ `langchain-mcp-adapters` |
| **MemFS versionado** (extensión Letta, no en paper) | ⚠️ `dulwich` (primitivas git) + abstracción propia | ⚠️ `dulwich` (primitivas git) + abstracción propia | ⚠️ `dulwich` + peleando con virtual FS de DeepAgents |
| **Esfuerzo estimado** | **ALTO** (2-4 sem., aún sin multi-agente/MCP) | **MEDIO** (2-4 sem.) | **MUY ALTO** (peor que LangGraph) |

**Lo que cambia al añadir Graphiti** (filas en verde donde antes había ⚠️ o ❌):

- **Recall Storage**: pasa de ⚠️/❌ a ✅ en los tres. Graphiti aporta knowledge graph **bi-temporal** + búsqueda híbrida (semántico + BM25 + graph traversal).
- **Archival Storage**: mejora cualitativa en LangGraph (de pgvector ingenuo a knowledge graph temporal con multi-hop) y resuelve el gap de DeepAgents.
- **Persistencia entre sesiones**: Graphiti persiste lo que él gestiona, lo que reduce el código de serialización a escribir.

**Lo que cambia al añadir librerías auxiliares específicas**:

- **APScheduler 3.x**: resuelve completamente eventos wall-clock sin pagar LangGraph Platform.
- **LangChain Middleware** (`before_model` / `after_model`): habilita Queue Manager con umbrales y eventos por iteraciones con ~10 líneas.
- **`litellm.token_counter`**: contador de tokens unificado para 100+ modelos (cross-provider).
- **`langmem.SummarizationNode`**: resumen recursivo con LLM separado al 80% (con bugs activos en tool calls).
- **`instructor`**: structured outputs para tipar `request_heartbeat` como tool de Pydantic.
- **`dulwich`**: Git puro Python para construir MemFS versionado (solo primitivas, sin abstracción de alto nivel).
- **Pydantic custom (~50 líneas)**: la solución más limpia para Core Memory con bloques etiquetados - más sencillo que cualquier librería externa.

**Lo que NO cambia ni con Graphiti ni con librerías auxiliares**:

- La **lógica de orquestación** del Queue Manager (cuándo cruzar umbral, qué expulsar, cuándo regenerar resumen) sigue siendo código de autor.
- El **edge condicional de `request_heartbeat`** sigue siendo manual (instructor solo tipa el tool).
- La **abstracción "memory filesystem for agents"** sobre dulwich no existe - hay que construirla.
- Los **bugs de langmem con tool calls** (#118, #111, #126) requieren parches o workarounds.

### Librerías auxiliares recomendadas (resumen por componente)

| # | Componente | Librería clave | Cobertura |
|---|---|---|---|
| 1 | Queue Manager doble umbral | `litellm.token_counter` + LangChain Middleware | Cuenta tokens y dispara hooks; lógica de umbrales propia |
| 2 | Resumen recursivo LLM separado | `langmem.SummarizationNode` | 80% del componente; **bugs activos con tool calls** |
| 3 | `request_heartbeat` | `instructor` + edge condicional propio | Tool tipado; orquestación manual |
| 4 | Wall-clock scheduling | **APScheduler 3.x** | Cobertura completa; estable, sin dependencias |
| 5 | Eventos por iteraciones | **LangChain Middleware** | ~10 líneas con contador en estado |
| 6 | MemFS versionado | `dulwich` | Primitivas git; abstracción de alto nivel propia |
| 7 | Core Memory bloques editables | **Pydantic custom (~50 líneas)** | Más limpio que cualquier librería externa |

⚠️ **Nota sobre versiones**: APScheduler 4.0 está en alpha y NO se recomienda para producción - usar la 3.x estable.

### Impacto en el esfuerzo estimado

Con todas estas piezas combinadas, el esfuerzo de implementación de MemGPT sobre LangGraph baja de **MEDIO-ALTO (3-6 semanas)** a **MEDIO (2-4 semanas)**. El ahorro real viene de:

- No construir el scheduler desde cero (APScheduler).
- No reinventar el resumen incremental (langmem, con caveats).
- No escribir contadores de tokens cross-provider (litellm).
- Los hooks de LangChain Middleware reducen drásticamente el boilerplate.

El trabajo de autor que **sigue siendo necesario**:

- Lógica de umbrales del Queue Manager.
- Capa de abstracción sobre dulwich para MemFS.
- Workarounds para los bugs de langmem.

Conclusión actualizada: **Graphiti nivela la capa de memoria, las librerías auxiliares cubren la mitad del runtime, pero las diferencias críticas siguen estando en el runtime** (StateGraph de LangGraph vs primitivas planas de LangChain core vs harness de DeepAgents). LangGraph sigue siendo la mejor base.

### LangChain core: descartado

- Las clases de memoria (`ConversationBufferMemory`, etc.) están **deprecated desde v0.3.1** y se eliminarán en v1.0.
- El propio `lang-memgpt` oficial de LangChain AI usa **LangGraph**, no LangChain core.
- Usar LangChain core hoy equivaldría a **reinventar LangGraph manualmente**. No tiene sentido.

### DeepAgents: descartado

- Está diseñado como **coding/task agent**, no como framework de memoria.
- Su filesystem virtual **no es MemFS** (sin versionado, sin branching, sin historial).
- Sus subagentes son **efímeros y aislados** - incompatible con multi-agente persistente al estilo Letta.
- Tiene **~20x más overhead de tokens** que LangGraph puro.
- Hay que **luchar contra sus abstracciones** (middleware stack, AGENTS.md, virtual FS) en lugar de construir sobre ellas.

### LangGraph: la opción ganadora

Es la base **más cercana al modelo del paper** y el ecosistema oficial más sólido:

- ✅ `StateGraph` mapea directamente al "agent loop" de MemGPT (`load_context → agent → tools`).
- ✅ `PostgresStore` con pgvector + HNSW **nativo** cubre Archival Storage.
- ✅ `PostgresSaver` cubre Recall Storage y persistencia entre sesiones.
- ✅ `langmem.short_term.SummarizationNode` es lo más cercano al slot 0 recursivo.
- ✅ Patrón de referencia oficial: [`langchain-ai/lang-memgpt`](https://github.com/langchain-ai/lang-memgpt) (simplificado, sin Queue Manager completo).
- ✅ Multi-agente y MCP de primera clase.

#### Lo que hay que construir a mano sobre LangGraph

1. **Queue Manager** con doble umbral (70/100%) + Memory Pressure Alert.
2. **Bloques etiquetados** de Core Memory (estado tipado + tools `core_memory_append/replace` que devuelven `Command(update={...})`).
3. **Router de `request_heartbeat`** como edge condicional en el StateGraph.
4. **`conversation_search`** como nodo que consulta el checkpointer/Store por similitud.
5. **Scheduler externo** para wall-clock (APScheduler) si no usas LangGraph Platform.
6. **MemFS versionado** si lo quieres (puedes prescindir de él en una primera versión).

#### Repos y SDKs útiles

- [`langchain-ai/lang-memgpt`](https://github.com/langchain-ai/lang-memgpt) - implementación de referencia oficial sobre LangGraph (simplificada).
- [`langchain-ai/langmem`](https://github.com/langchain-ai/langmem) - SDK de LangChain para memoria long-term (incluye `SummarizationNode`).
- [LangGraph Persistence Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/persistence.md).
- [Semantic Search for LangGraph Memory](https://blog.langchain.com/semantic-search-for-langgraph-memory/) - anuncio del semantic search nativo en `BaseStore`.

### Frameworks fuera del ecosistema LangChain

¿Hay algo mejor que LangGraph fuera del mundo LangChain? Evaluación de los principales frameworks generalistas y librerías de memoria especializadas.

#### Frameworks de agente generalistas

| Framework | Veredicto resumido |
|---|---|
| **OpenAI Agents SDK** | Equivalente. Sesiones persistentes (SQLite/Postgres/Redis) + MCP nativo + multi-agente. Atado a infra OpenAI. Sin umbral dual ni FIFO con presión. |
| **Microsoft Agent Framework 1.0** (fusión AutoGen + Semantic Kernel) | Maduro y con memoria pluggable (Mem0, Redis, Neo4j, Cosmos DB). Sesgado a Azure/C#, menos ágil para Python custom. |
| **AutoGen standalone** | Bueno para multi-agente conversacional. Sin context pressure, sin FIFO, sin core memory estructurada. |
| **CrewAI** | 4 tipos de memoria, pero **context bleeding entre usuarios** y arquitectura "crew" no encaja con loop de agente único. **Peor base que LangGraph.** |
| **Pydantic AI** | Limpio y type-safe (`MemoryTool`, `FileSearchTool`). Solo scaffolding de agente - sin gestión de ventana de contexto. |
| **Haystack** | Pipelines explícitos, ideal para RAG production-grade. No es agent loop stateful. |
| **DSPy** | Optimización de prompts, no agente stateful. **Descartable** para este caso. |
| **Smolagents** | Ligero (~1000 LoC). Bueno para prototipos rápidos, no como base completa. |

#### Librerías especializadas de memoria (no son agent loop)

Estas librerías cubren **solo la capa de memoria**, no son frameworks de agente. Se usan **encima** de un runtime:

| Librería | Lo que aporta |
|---|---|
| **mem0** | **La capa de memoria más robusta del ecosistema**. Híbrido vector + grafo (Neo4j/Kuzu) + key-value, BM25 + semántico + entity linking. **-91% latencia p95**, **-90% tokens** vs full-context. Paper: [arXiv:2504.19413](https://arxiv.org/abs/2504.19413). |
| **Zep / Graphiti** | Knowledge graph **temporal**. **Supera a MemGPT en DMR (94.8% vs 93.4%)**. Hybrid search (semántico + BM25 + graph traversal), p95 ~300ms. Paper: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956). Repo: [getzep/graphiti](https://github.com/getzep/graphiti). |
| **Cognee** | Pipeline ECL (Extract, Cognify, Load) con knowledge graph desde 38+ fuentes. MCP nativo. Útil como backend de Archival, no como runtime. |

#### Caso especial: LlamaIndex

LlamaIndex introdujo en 2025 **"memory blocks" con flush automático** cuando se supera `chat_history_token_ratio` (default 0.7) - es lo más cercano conceptualmente al FIFO con umbral de MemGPT fuera de Letta. Pero es menos granular: no hay warning threshold separado, no hay Memory Pressure Alert, y el resumen recursivo no usa una llamada LLM separada del agent loop. [Docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/).

### Conclusión: LangGraph + Graphiti como pareja óptima

**Ningún framework fuera de LangChain es claramente superior a LangGraph como base** del agent loop. Pero la investigación reveló algo importante: **las librerías especializadas de memoria son objetivamente mejores que pgvector "a pelo"** para Recall/Archival.

La combinación óptima en 2026 para construir MemGPT desde cero **no es LangGraph solo**, sino:

> **LangGraph (runtime + agent loop) + Graphiti (backend de Recall/Archival Storage)**

Cada uno cubre lo que mejor sabe hacer:

- **LangGraph**: agent loop, Queue Manager, core memory blocks, router de `request_heartbeat`, persistencia de sesión vía checkpointer.
- **Graphiti**: extracción de hechos, knowledge graph **bi-temporal**, búsqueda híbrida (semántico + BM25 + graph traversal).

Ventajas concretas frente a LangGraph + pgvector ingenuo:

- **+1.4 puntos sobre MemGPT en DMR** (94.8% vs 93.4%).
- **p95 ~300ms** en búsquedas.
- Razonamiento temporal nativo: capacidad de modelar hechos que evolucionan en el tiempo (clave para el ejemplo "novio → ex-novio" de la Figura 4).
- Queries multi-hop sobre el grafo (útil para tareas tipo Nested KV).

> **Alternativa**: si el caso de uso es más simple (chatbot de soporte, asistente personal sin razonamiento temporal complejo), **mem0** es el sustituto natural - más simple de integrar, con métricas de eficiencia (-91% latencia p95, -90% tokens) aunque sin la riqueza temporal de Graphiti.

### Plan recomendado de implementación

**Plan A (pedagógico, recomendado)**:

1. Forkear [`lang-memgpt`](https://github.com/langchain-ai/lang-memgpt) como esqueleto inicial.
2. Reemplazar Pinecone por `PostgresStore` con pgvector + HNSW (versión ingenua).
3. Añadir el Queue Manager (umbrales + pressure alert) como nodos extra.
4. Añadir `conversation_search` y bloques etiquetados de Core Memory.
5. Iterar hasta replicar **Nested KV** (el benchmark más simple del paper).
6. Avanzar a DMR (MSC) y finalmente Document QA.
7. **Mejora opcional**: una vez funcionando, **sustituir el backend de almacenamiento por Graphiti** y medir mejoras en precisión sobre DMR (objetivo: superar el 93.4% del paper alcanzando el 94.8% reportado por Graphiti).

**Plan B (más rápido)**:

- Usar **Letta directamente** y limitarte a estudiar `letta/agent.py` para entender la arquitectura. Pierdes el ejercicio de implementación pero ganas semanas.

### Veredicto

| Si quieres... | Usa |
|---|---|
| Producción sin escribir código de memoria | **Letta** directamente |
| Construir desde cero por aprendizaje (versión simple) | **LangGraph** + `lang-memgpt` + pgvector |
| Construir desde cero apuntando a producción | **LangGraph** + **Graphiti** como backend |
| Mejor benchmark DMR (superar al paper) | LangGraph + **Graphiti** (94.8% vs 93.4%) |
| Caso de uso simple sin razonamiento temporal | LangGraph + **mem0** (alternativa más simple a Graphiti) |
| Coding/task agent (no MemGPT) | **DeepAgents** sigue siendo válido |
| Algo "rápido" en LangChain core | No existe - **descartar** |
