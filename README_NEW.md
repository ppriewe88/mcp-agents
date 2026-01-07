Projektübersicht – Agenten-Architektur & Streaming-Ansatz

1. Zielbild (High-Level)

Das Projekt implementiert ein agentenbasiertes Backend auf Basis von LangChain / LangGraph, das:

Agenten konfigurationsgetrieben erstellt (Factory-Pattern),

Agenten sowohl direkt als auch als Tool für andere Agenten einsetzen kann,

echtes Streaming (Token-/Event-basiert) zum Frontend ermöglicht,

und langfristig strukturierte, typisierte Stream-Events (z. B. NDJSON) an das Frontend liefert.

Der Fokus liegt nicht auf schnellen Prototypen, sondern auf:

klarer Trennung von Verantwortung,

stabiler Streaming-Semantik,

und kontrollierbarer Agenten-Orchestrierung.

1. Zentrale Konzepte & Domänenobjekte
2.1 CompleteAgent

Was es ist:
Ein reines Konfigurationsobjekt (keine Runtime-Logik).

Enthält:

config: AgentConfig (Systemprompt, Loop-Kontrolle, Validation etc.)

tool_schemas: List[ToolSchema] (klassische Tools, z. B. MCP-basiert)

agents_as_tools: List[Any] (Agenten, die als Tool fungieren)

description

Wichtig:
CompleteAgent beschreibt was ein Agent ist – nicht wie er ausgeführt wird.

2.2 AgentConfig

Was es ist:
Verhaltens- und Kontrollkonfiguration für einen Agenten.

Beispiele:

System Prompt

only_one_model_call

Toolcall-Limits

Validation-Prompts

Final-Prompt-Overrides

Diese Konfiguration wird später von der Factory in Middleware übersetzt.

2.3 ConfiguredAgent

Was es ist:
Der laufzeitfähige Wrapper um einen kompilierten LangGraph-Agenten.

Verantwortung:

hält den kompilierten CompiledStateGraph

hält den initial_state (CustomStateShared)

bietet eine einheitliche Runtime-Schnittstelle

Exponierte Methoden:

run(query: str) → synchron / non-streaming

outer_astream(query: str) → Streaming nach außen (Frontend)

Wichtig:

ConfiguredAgent kennt nur den äußeren Stream

kein inneres Streaming, keine Tool-Hüllen

bewusst einfach gehalten

1. Factory-Ansatz
3.1 AgentFactory

Zentrale Aufgabe:
Erzeugt aus einem CompleteAgent einen lauffähigen ConfiguredAgent.

Pipeline:

Tools laden

MCP-basierte Tools aus tool_schemas

Agent-als-Tool aus agents_as_tools

Middleware zusammenbauen

Logging

Loop-Kontrolle

Toolcall-Limits

Validation

LangChain-Agent bauen

create_agent(...)

In ConfiguredAgent wrappen

Ergebnis:
Ein vollständig konfigurierter, isolierter Agent mit definierter Runtime-API.

1. Streaming-Architektur (aktueller Stand)
4.1 Outer Stream (ConfiguredAgent.outer_astream)

Zweck:
Streaming zum Frontend.

Verwendet:

self.agent.astream(
    extended_state,
    stream_mode=["messages", "updates", "custom"]
)

Verarbeitet:

updates:

Toolcall requested (AIMessage mit tool_calls)

Toolcall result (ToolMessage)

Final Answer (validated_agent_output)

custom:

aktuell nur Debug-Prints

kommen von inneren Agenten (Agent-as-Tool)

Ausgabe:

AsyncGenerator[bytes]

aktuell reines Text-Streaming (text/plain)

4.2 Agent-as-Tool (innerer Agent)

Umsetzung:

Ein ConfiguredAgent wird als Tool via @tool eingebunden.

Die Tool-Hülle:

ruft inner_agent.agent.astream(...)

verarbeitet nur updates

filtert auf:

toolcall_requested

toolcall_result

final_answer

emittiert diese Events via:

writer = get_stream_writer()
writer({...})  # custom event

Wichtiges Architekturprinzip:

Kein inneres Streaming im ConfiguredAgent

Inneres Streaming lebt exklusiv in der Tool-Hülle

get_stream_writer() funktioniert, da Tool im Runnable-Kontext läuft

1. API-Integration (FastAPI)
5.1 Aktueller Endpunkt
@app.post("/stream-test")

Modi:

simulated_stream

klassischer Run

Ausgabe über artificial_stream

true_stream

echtes Streaming

StreamingResponse(agent.outer_astream(...))

Medientyp:

media_type="text/plain"

Status:

stabil

geeignet für erste Frontend-Tests

noch untypisierte Events

1. Aktueller Gesamtstatus
Was funktioniert:

Factory & Agent-Zusammenbau

Outer Streaming

Agent-as-Tool

Inner Agent streamt Ereignisse live in den Outer Stream

custom-Events kommen zuverlässig an

Was bewusst noch nicht gemacht ist:

Visualisierung innerer Toolcalls im Frontend

Typisierung der Stream-Events

NDJSON / SSE-Format

Konsolidierte Event-Semantik (outer + inner)

1. Nächste geplante Schritte (fixiert)
Schritt 1 – Custom Events sichtbar machen

custom-Events im outer_astream nicht nur printen

sondern:

wie outer Toolcalls behandeln

Marker ausgeben

später strukturiert weiterreichen

Schritt 2 – Event-Typisierung

Alle Events, die ans Frontend gehen, werden:

nicht mehr zu String

sondern zu strukturierten Objekten

z. B.:

{"type": "toolcall_requested", "agent": "inner", "tool": "add"}

Schritt 3 – Streaming-Format

Umstellung von text/plain auf:

NDJSON oder

Server-Sent Events (SSE)

Frontend kann Events differenziert darstellen:

Toolcall gestartet

Toolcall beendet

Final Answer

Abort / Validation

1. Mentales Modell (wichtig für Weiterarbeit)

Factory baut Agenten → statisch

ConfiguredAgent führt Agenten aus → Runtime

Tool-Hülle = Brücke zwischen Agenten → Orchestrierung

Outer Stream = einzige Quelle für Frontend-Events

Inner Stream = reduziert, gefiltert, weitergereicht

Wenn du mir künftig einfach sagst:

„Wir sind bei dem Agent-as-Tool Streaming-Projekt, Factory + ConfiguredAgent, inner stream über Tool-Hülle, outer_astream streamt ans Frontend – wir wollen jetzt die custom Events typisieren und visualisieren“
