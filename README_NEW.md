Projektübersicht – Agenten-Architektur & Streaming-Ansatz (aktualisiert)

Zielbild (High-Level)

Das Projekt implementiert ein agentenbasiertes Backend auf Basis von LangChain / LangGraph, das:

Agenten konfigurationsgetrieben erstellt (Factory-Pattern),

Agenten sowohl direkt als auch als Tool für andere Agenten einsetzen kann,

echtes Streaming (Token-/Event-basiert) zum Frontend ermöglicht,

und strukturierte, typisierte Stream-Events (NDJSON) an das Frontend liefert.

Der Fokus liegt auf:

klarer Trennung von Verantwortung,

stabiler Streaming-Semantik,

kontrollierbarer Agenten-Orchestrierung,

und einer Streaming-Schnittstelle, die sich schrittweise um strukturierte Payloads erweitern lässt, ohne das Protokoll zu brechen.

Zentrale Konzepte & Domänenobjekte

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

2.3 RunnableAgent (vormals “ConfiguredAgent” in der Doku)

Was es ist:
Der laufzeitfähige Wrapper um einen kompilierten LangGraph-Agenten.

Verantwortung:

hält den kompilierten CompiledStateGraph,

hält den initial_state (CustomStateShared),

konstruiert aus Frontend-Chatnachrichten LangChain-Threads,

bietet eine einheitliche Runtime-Schnittstelle für non-streaming und streaming,

normalisiert alle Streaming-Ausgaben in NDJSON-Chunks.

Exponierte Methoden:

run(messages: List[ChatMessage]) → non-streaming (ainvoke)

outer_astream(messages: List[ChatMessage]) → Streaming nach außen (Frontend)

Wichtig (neuer Stand):
RunnableAgent kapselt nun die gesamte Streaming-Normalisierung (outer + inner), indem interne Ereignisse (custom + updates) in ein typisiertes Chunk-Modell überführt und anschließend einheitlich serialisiert werden.

Factory-Ansatz

3.1 AgentFactory

Zentrale Aufgabe:
Erzeugt aus einem CompleteAgent einen lauffähigen RunnableAgent.

Pipeline:

Tools laden

MCP-basierte Tools aus tool_schemas

Agent-als-Tool aus agents_as_tools

Middleware zusammenbauen

Logging

Loop-Kontrolle

Toolcall-Limits

Validation

LangChain/LangGraph-Agent bauen

create_agent(...)

In RunnableAgent wrappen

Ergebnis: vollständig konfigurierter Agent mit definierter Runtime-API

Streaming-Architektur (aktueller Stand)

4.1 Streaming-Format: NDJSON als Transportprotokoll

Der Endpoint /stream-test liefert eine StreamingResponse mit:

media_type="application/x-ndjson"

jedes Streaming-Element ist ein JSON-Objekt pro Zeile (NDJSON)

der Stream ist robust gegenüber Chunk-Grenzen, da das Frontend zeilenbasiert puffert und JSON.parse pro Zeile ausführt

4.2 Pydantic-Datenmodell: StreamChunk (neu)

Es wurde ein explizites Pydantic-Modell eingeführt, das sowohl inneres als auch äußeres Streaming in eine einheitliche Ereignisrepräsentation überführt:

StreamLevel: OUTER ("outer_agent") vs INNER ("inner_agent")

StreamEvent: START, TOOL_REQUEST, TOOL_RESULT, FINAL, ABORTED

StreamChunk: level, event, agent_name und optionale Felder (tool_name, toolcall_id, data, final_answer, aborted, abortion_reason, etc.)

Damit ist die Grundlage gelegt, pro Ereignis ein strukturierteres Payload zu transportieren (insbesondere bei Tool-Ergebnissen).

4.3 Outer Stream: RunnableAgent.outer_astream (neu strukturiert)

Zweck:
Streaming zum Frontend als NDJSON, basierend auf:

self.agent.astream(
extended_state,
stream_mode=["messages", "updates", "custom"]
)

Verarbeitungslogik:

messages: werden unterdrückt (kein Streaming dieser Low-Level Token-Chunks)

custom: repräsentiert weitergereichte StreamChunks aus inneren Agenten (Agent-as-Tool), wird validiert und als StreamChunk verarbeitet

updates: repräsentiert State-/Middleware-Updates des Outer-Agenten, wird extrahiert, in StreamChunks übersetzt und verarbeitet

Die Verarbeitung ist in zwei Handler ausgelagert:

\_handle_subagent_stream(data): verarbeitet custom-Events (inner agent)

\_handle_agent_stream(data, emitted_toolcall_ids): verarbeitet updates (outer agent)

Beide Handler arbeiten nicht mehr direkt mit untypisierten Strings, sondern erzeugen/validieren StreamChunks und delegieren das Senden an den zentralen Emitter.

4.4 Inner Stream: Agent-as-Tool (innerer Agent) als Producer von StreamChunks

Ein innerer Agent wird als Tool-Hülle eingebunden. Diese Tool-Hülle:

ruft subagent.agent.astream(..., stream_mode=["updates","messages","custom"])

verarbeitet nur updates (messages werden unterdrückt)

extrahiert Tool Requests, Tool Results, Final Output, Aborts

erstellt dafür StreamChunk-Objekte (level=INNER, event=...)

sendet diese über get_stream_writer() als custom-Events in den Outer Stream

Prinzip:
Inneres Streaming bleibt in der Tool-Hülle (Orchestrierungsgrenze), aber die Event-Semantik ist nun identisch zum Outer Stream (gleiche StreamChunk-Typen).

4.5 Zentrale Emission: \_emit_chunk_ndjson (neu, zentraler Output-Contract)

RunnableAgent besitzt eine zentrale Emissionsmethode:

\_emit_chunk_ndjson(chunk: StreamChunk) -> AsyncGenerator[bytes, None]

Diese Methode implementiert die verbindliche Mapping-Logik von (level, event) → NDJSON-Ausgabe:

Aktuelle Regeln (Stand heute):

A) TOOL_RESULT (inner + outer)

NDJSON: {"level": "<outer_agent|inner_agent>", "type": "tool_results", "data": chunk.data}

data ist für Tool-Ergebnisse vorgesehen und soll zukünftig strukturiert (dict/list) sein

B) FINAL

OUTER FINAL:

NDJSON: type="text_final"

Ausgabe erfolgt über artificial_stream(text), d. h. die finale Antwort wird in Textfragmenten gestreamt (laufender Effekt im Frontend)

INNER FINAL:

NDJSON: type="text_final"

Ausgabe erfolgt als ein einzelnes Text-Element (kein artificial_stream), da inneres Final primär als Zwischenstand/Debug dient und im Frontend aktuell nicht in die finale Messagebox geroutet wird

C) START / TOOL_REQUEST / ABORTED

NDJSON: type="text_step"

data ist ein Marker-Text, z. B.:

"[outer_agent] CALLING TOOL: ..."

"[inner_agent] ABORTED: ..."

D) Uncovered Events

\_emit_chunk_ndjson wirft ValueError, um sicherzustellen, dass neue Events explizit abgebildet werden.

API-Integration (FastAPI)

5.1 Endpunkt /stream-test

@app.post("/stream-test")

nimmt StreamAgentRequest entgegen

assembliert Agent (Factory) bzw. nutzt Test-Agent

liefert StreamingResponse(agent.outer_astream(messages), media_type="application/x-ndjson")

Damit ist das Backend-Protokoll stabil:

Frontend erhält NDJSON und kann anhand von type/level differenziert handeln.

Aktueller Gesamtstatus

Was funktioniert (Stand heute):

Factory & Agent-Zusammenbau

Outer Streaming via NDJSON

Agent-as-Tool mit innerem Streaming über get_stream_writer()

Konsolidiertes Event-Modell StreamChunk (Pydantic) für inner und outer

Ausgelagerte Stream-Handler:

\_handle_subagent_stream: validiert/normalisiert custom-Events zu StreamChunks und emittiert

\_handle_agent_stream: übersetzt updates zu StreamChunks (\_extract_agent_chunks) und emittiert

Zentraler NDJSON-Emitter \_emit_chunk_ndjson mit klaren Regeln:

tool_results als strukturierter Kanal

final_text (outer) als künstlich gestreamte Textfragmente

Was bewusst noch nicht gemacht ist:

echte strukturierte Tool-Payloads (Tabellenmodell) als stabiles Schema

Konsolidierung der inneren vs äußeren FINAL-Typen (derzeit beide type="text_final", Differenzierung erfolgt über level)

standardisierte Fehler-/Abbruch-Payloads mit separatem type (z. B. type="error" statt text_step Marker)

Sequencing (seq) / Timestamp in Chunks

Tests für Chunk-Grenzen und Payload-Validität

Mentales Modell (wichtig für Weiterarbeit)

Factory baut Agenten (statisch)

RunnableAgent führt Agenten aus (Runtime)

Tool-Hülle ist die Brücke zwischen Agenten (Orchestrierung + inner streaming)

Outer Stream ist die einzige Quelle für Frontend-Events

Inner Stream wird gefiltert und als custom-Events weitergereicht, aber semantisch identisch (StreamChunk)

NDJSON ist der stabile Transport; type + level steuern das Rendering im Frontend

Nächste Schritte (Backend) – Fokus: Toolcall data als dict (strukturierte Payloads)

Ziel:
Tool-Ergebnisse sollen nicht mehr als String in data landen, sondern als JSON-Objekt (dict/list) mit definiertem Schema, sodass das Frontend Tabellen/Artefakte sicher darstellen kann.

Schritt 1 – Sicherstellen, dass ToolMessage-Inhalte JSON-serialisierbar sind

Aktuell: chunk.data = last.content (kann str, list, dict oder komplex sein)

Maßnahme:

vor dem Einfüllen in StreamChunk.data prüfen/normalisieren:

wenn Pydantic-Modell: model_dump(mode="json")

wenn LangChain Content-Struktur: in primitives überführen (dict/list/str)

fallback: str(...) nur, wenn keine Struktur möglich ist

Schritt 2 – Tool-Result-Schema definieren (MVP Tabellenmodell)

Einführen eines einfachen, stabilen Schemas z. B.:

{"kind": "table", "columns": [...], "rows": [...]}

optional: {"kind": "file_ref", "filename": "...", "mime": "...", "bytes_b64": "..."} (später)

StreamChunk.data für TOOL_RESULT enthält dann immer:

dict mit key "kind" und den notwendigen Feldern

Schritt 3 – Pydantic-Union für ToolResultPayload

data: Optional[Any] ersetzen durch Optional[ToolResultPayload]

ToolResultPayload als Union:

TablePayload

TextPayload

GenericJsonPayload (für Übergangsphase)

Schritt 4 – Output-Typen weiter stabilisieren

optional: statt type="tool_results" künftig differenzierte types:

type="tool_result_table"

type="tool_result_json"

alternativ (MVP): type="tool_results", Differenzierung über data.kind

Schritt 5 – Tests

Unit-/Integrationtests:

TOOL_RESULT enthält nur JSON-serialisierbare Strukturen

NDJSON-Format bleibt korrekt (eine Zeile = ein JSON-Objekt)

Reihenfolge der Chunks entspricht der Ausführungsreihenfolge (custom + updates)

Final wird nur vom Outer Agent gestreamt (level=outer_agent)

To-dos (konsolidiert)

StreamChunk.data für TOOL_RESULT systematisch normalisieren (JSON-serialisierbar erzwingen).

Einfaches Tabellen-/Artefakt-Schema definieren und in Tools/Tool-Hülle standardisieren.

Pydantic-Modelle/Enums für Payloads ergänzen (Union), um Typsicherheit über die gesamte Pipeline zu bekommen.

Optional: seq/timestamp in StreamChunk aufnehmen.

Einheitliche error/aborted Chunk-Typen definieren (nicht nur Marker-Text).

Tests für NDJSON-Streaming (Chunk-Grenzen, Typen, Reihenfolge, Payload-Serialisierbarkeit) hinzufügen.
