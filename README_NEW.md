Backend

Wir definieren ein kleines, stabiles NDJSON-Event-Schema als Union mit einem Pflichtfeld type und einem payload (z. B. { type: "text_message" | "toolcall_request" | "toolcall_result", payload: ... }), optional ergänzt um id, ts, agent_name zur Debugbarkeit.

In ConfiguredAgent.astream() streamen wir ausschließlich updates und emittieren für Toolcalls (AIMessage.tool_calls) jeweils ein toolcall_request-Event (inkl. toolName, toolCallId, optional argsPreview), und für ToolMessages ein toolcall_result-Event (inkl. toolName, toolCallId, sowie data/text als Ergebnis).

Sobald validated_agent_output gesetzt ist, streamen wir die finale Antwort künstlich wie heute, aber als Sequenz von text_message-Events (je Chunk ein NDJSON-Objekt), und beenden danach den Generator.

Der Endpoint /stream-test stellt media_type="application/x-ndjson" ein, sodass jede Zeile ein vollständiges JSON-Event ist (kein gemischtes Plaintext-/Marker-Protokoll mehr).

Frontend

invokeAgent bleibt strukturell gleich, aber statt rawChunk direkt durchzureichen, bauen wir im Reader einen kleinen Zeilen-Puffer (String-Buffer) und splitten nach \n, um komplette NDJSON-Zeilen zu extrahieren und zu JSON.parsen (ein Event pro Zeile).

streamControl wird von chunk: string auf event: StreamEvent umgestellt (oder bekommt beides), filtert/normalisiert Events und gibt typisiert zurück, z. B. { appendText?: string, toolcallRequest?: {...}, toolcallResult?: {...} }.

Im sendMessage-Callsite ersetzen wir renderChunk(appendText) durch einen minimalen Dispatcher: text_message → wie bisher an aiId.content anhängen; toolcall_request/toolcall_result → in eine separate UI-Struktur schreiben (z. B. toolEvents im Chat-State oder als meta-Feld in der AI-Message), sodass später eine Tabelle/Komponente daraus rendern kann.

Minimal-invasiv bleibt, dass der bestehende Textpfad unverändert ist (nur NDJSON-Decoding + Switch auf type), während Toolcall-Events zunächst lediglich gesammelt/geloggt werden können und erst später in dedizierte Komponenten (z. B. „Tool Activity“) gerendert werden.
