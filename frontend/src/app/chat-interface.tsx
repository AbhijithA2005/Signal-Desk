"use client";

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  FileText,
  Image as ImageIcon,
  MessageSquare,
  Plus,
  Search,
  Send,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { PredictiveArcCanvas } from "@designcodeio/threeui";
import "@designcodeio/threeui/style.css";

const API_URL = "/api";

type SearchResult = {
  chunk_id: string;
  document_id: string;
  text: string;
  score?: number | null;
  reranker_score?: number | null;
  citation: string;
};

type Source = {
  document_id: string;
  chunk_id: string;
  page_label: string;
  chunk_index: number;
  score?: number | null;
};

type AskResponse = {
  query: string;
  answer: string;
  decision: "generate" | "abstain";
  gate_score?: number | null;
  sources: Source[];
  retrieval: SearchResult[];
};

type NotesResponse = Omit<AskResponse, "answer"> & { notes: string };

type ImageAnalysisResponse = {
  query: string;
  analysis: string;
  notice: string;
};

type SearchResponse = {
  query: string;
  results: SearchResult[];
};

type StreamEvent =
  | { type: "metadata"; payload: AskResponse | NotesResponse | ImageAnalysisResponse }
  | { type: "token"; text: string }
  | { type: "error"; message: string }
  | { type: "done" };

type ConversationMode = "general" | "ask" | "notes" | "search";
type TurnMode = ConversationMode | "image";

type ConversationTurn = {
  id: string;
  question: string;
  mode: TurnMode;
  streaming?: boolean;
  response?: AskResponse;
  imageAnalysis?: ImageAnalysisResponse;
  results?: SearchResult[];
  error?: string;
};

type UploadedDocument = {
  document_id: string;
  filename: string;
  sha256: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  created_at: string;
  duplicate?: boolean;
};

function displayDocumentName(
  documentId: string,
  uploadedDocuments: UploadedDocument[],
) {
  return (
    uploadedDocuments.find((document) => document.document_id === documentId)
      ?.filename ?? documentId
  );
}

function citationWithoutDocumentId(result: SearchResult) {
  const prefix = `${result.document_id}, `;
  return result.citation.startsWith(prefix)
    ? result.citation.slice(prefix.length)
    : result.citation;
}

function StreamedMarkdown({ text, streaming }: { text: string; streaming: boolean }) {
  return (
    <div className="answer-markdown" aria-live="off">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      {streaming && <span className="answer-caret" aria-hidden="true" />}
    </div>
  );
}

export default function ChatInterface() {
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<ConversationMode>("general");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [documentError, setDocumentError] = useState("");
  const endOfConversation = useRef<HTMLDivElement>(null);
  const queryTextareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    endOfConversation.current?.scrollIntoView({
      behavior: "auto",
      block: "end",
    });
  }, [turns, loading]);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/documents`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load uploaded documents.");
        return (await response.json()) as { documents: UploadedDocument[] };
      })
      .then((data) => {
        if (active) {
          setDocuments(data.documents);
          setSelectedDocumentIds(
            data.documents.map((document) => document.document_id),
          );
          setMode(data.documents.length > 0 ? "ask" : "general");
        }
      })
      .catch(() => {
        if (active) setDocumentError("Uploaded documents are unavailable.");
      });

    return () => {
      active = false;
    };
  }, []);

  async function consumeStream(
    response: Response,
    turnId: string,
    requestMode: TurnMode,
  ) {
    if (!response.body) throw new Error("The response stream is unavailable.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";

    function handleEvent(event: StreamEvent) {
      setTurns((current) =>
        current.map((turn) => {
          if (turn.id !== turnId) return turn;

          if (event.type === "metadata") {
            if (requestMode === "image") {
              return {
                ...turn,
                imageAnalysis: {
                  ...(event.payload as ImageAnalysisResponse),
                  analysis: "",
                },
                streaming: true,
              };
            }
            if (requestMode === "notes") {
              const notes = event.payload as NotesResponse;
              return {
                ...turn,
                response: { ...notes, answer: notes.notes },
                streaming: true,
              };
            }
            return {
              ...turn,
              response: event.payload as AskResponse,
              streaming: true,
            };
          }

          if (event.type === "token") {
            if (requestMode === "image" && turn.imageAnalysis) {
              return {
                ...turn,
                imageAnalysis: {
                  ...turn.imageAnalysis,
                  analysis: turn.imageAnalysis.analysis + event.text,
                },
              };
            }
            if (turn.response) {
              return {
                ...turn,
                response: {
                  ...turn.response,
                  answer: turn.response.answer + event.text,
                },
              };
            }
            return turn;
          }

          if (event.type === "error") {
            return { ...turn, error: event.message, streaming: false };
          }
          return { ...turn, streaming: false };
        }),
      );
    }

    while (true) {
      const { value, done } = await reader.read();
      buffered += decoder.decode(value, { stream: !done });
      let newline = buffered.indexOf("\n");
      while (newline !== -1) {
        const line = buffered.slice(0, newline);
        buffered = buffered.slice(newline + 1);
        if (line.trim()) handleEvent(JSON.parse(line) as StreamEvent);
        newline = buffered.indexOf("\n");
      }
      if (done) break;
    }
    if (buffered.trim()) handleEvent(JSON.parse(buffered) as StreamEvent);
  }

  async function askQuestion(
    event?: FormEvent<HTMLFormElement>,
    suggestedQuestion?: string,
  ) {
    event?.preventDefault();

    const cleanQuery = (suggestedQuestion ?? query).trim();
    const image = selectedImage;
    if ((!cleanQuery && !image) || loading) return;
    if (!image && mode !== "general" && selectedDocumentIds.length === 0) {
      setDocumentError("Choose or add sources before asking a question.");
      return;
    }

    const requestMode: TurnMode = image ? "image" : mode;
    const requestDocumentIds = selectedDocumentIds;
    const turnId = globalThis.crypto.randomUUID();
    setTurns((current) => [
      ...current,
      {
        id: turnId,
        question: cleanQuery || (image ? "Describe this image." : ""),
        mode: requestMode,
      },
    ]);
    setQuery("");
    if (queryTextareaRef.current) queryTextareaRef.current.style.height = "auto";
    setSelectedImage(null);
    setLoading(true);

    try {
      let response: Response;
      if (image) {
        const formData = new FormData();
        formData.append("file", image);
        formData.append("query", cleanQuery || "Describe this image.");
        response = await fetch(`${API_URL}/analyze-image/stream`, {
          method: "POST",
          body: formData,
        });
      } else {
        const endpoint =
          requestMode === "general"
            ? "chat/stream"
            : requestMode === "search"
              ? "search"
              : requestMode === "notes"
                ? "notes/stream"
                : "ask/stream";
        response = await fetch(`${API_URL}/${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: cleanQuery,
            ...(requestMode === "search" ? { top_k: 5 } : {}),
            ...(requestMode === "general"
              ? {}
              : { document_ids: requestDocumentIds }),
          }),
        });
      }

      if (!response.ok) {
        const errorData = (await response.json()) as { detail?: string };
        throw new Error(errorData.detail ?? "Request failed");
      }

      if (requestMode !== "search") {
        await consumeStream(response, turnId, requestMode);
        return;
      }

      const data = await response.json();
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? { ...turn, results: (data as SearchResponse).results }
            : turn,
        ),
      );
    } catch (error) {
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
              streaming: false,
                error:
                  error instanceof Error
                    ? error.message
                    : "Something went wrong while processing your question. Please try again.",
              }
            : turn,
        ),
      );
    } finally {
      setLoading(false);
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId ? { ...turn, streaming: false } : turn,
        ),
      );
    }
  }

  function startNewConversation() {
    if (loading) return;
    setTurns([]);
    setQuery("");
    if (queryTextareaRef.current) queryTextareaRef.current.style.height = "auto";
    setMode(documents.length > 0 ? "ask" : "general");
  }

  async function uploadFiles(event: ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    const files = Array.from(input.files ?? []);
    input.value = "";
    if (files.length === 0) return;

    setUploading(true);
    setDocumentError("");
    const uploaded: UploadedDocument[] = [];
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(`${API_URL}/documents`, {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail ?? `Could not upload ${file.name}.`);
        }
        uploaded.push(data as UploadedDocument);
      }

      setDocuments((current) => {
        const byId = new Map(current.map((document) => [document.document_id, document]));
        uploaded.forEach((document) => byId.set(document.document_id, document));
        return [...byId.values()];
      });
      setSelectedDocumentIds((current) => [
        ...new Set([
          ...current,
          ...uploaded.map((document) => document.document_id),
        ]),
      ]);
      setMode("ask");
    } catch (error) {
      setDocumentError(
        error instanceof Error ? error.message : "The PDF upload failed.",
      );
      try {
        const response = await fetch(`${API_URL}/documents`);
        if (response.ok) {
          const data = (await response.json()) as { documents: UploadedDocument[] };
          setDocuments(data.documents);
        }
      } catch {
        // Keep the upload error visible if the document list cannot refresh.
      }
    } finally {
      setUploading(false);
    }
  }

  function toggleDocument(documentId: string, checked: boolean) {
    setSelectedDocumentIds((current) => {
      return checked
        ? [...new Set([...current, documentId])]
        : current.filter((id) => id !== documentId);
    });
  }

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    setSelectedImage(event.currentTarget.files?.[0] ?? null);
    event.currentTarget.value = "";
  }

  async function deleteDocument(document: UploadedDocument) {
    if (!window.confirm(`Remove ${document.filename} from this workspace?`)) return;

    setDocumentError("");
    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(document.document_id)}`,
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error("Could not remove the uploaded document.");
      setDocuments((current) =>
        current.filter((item) => item.document_id !== document.document_id),
      );
      setSelectedDocumentIds((current) =>
        current.filter((id) => id !== document.document_id),
      );
    } catch (error) {
      setDocumentError(
        error instanceof Error ? error.message : "Could not remove the document.",
      );
    }
  }

  const hasSelectedSources = selectedDocumentIds.length > 0;
  const activeTurn = turns[turns.length - 1];
  const hasStreamedText = turns.some(
    (turn) =>
      turn.streaming &&
      Boolean(turn.response?.answer || turn.imageAnalysis?.analysis),
  );
  const processingMessage =
    activeTurn?.mode === "image"
      ? "Image received · analyzing"
      : activeTurn?.mode === "general"
        ? "Thinking"
        : activeTurn?.streaming
          ? "Evidence found · drafting response"
          : "Finding relevant evidence";

  return (
    <main className="chat-app">
      <PredictiveArcCanvas
        className="rag-arc-background"
        variant="data-pixel"
        mode="dark"
        speed={1.0}
        hue={0}
        saturation={1.0}
        brightness={1.0}
      />
      <div className="rag-arc-scrim" aria-hidden="true" />
      <div className="chat-shell">
        <header className="chat-header">
          <Link className="brand" href="/" aria-label="Signal Desk home">
            <span className="brand-mark" aria-hidden="true">
              <BookOpen size={18} strokeWidth={1.8} />
            </span>
            <span>Signal Desk</span>
          </Link>

          {turns.length > 0 && (
            <button
              className="new-chat-button"
              type="button"
              onClick={startNewConversation}
              disabled={loading}
              title="Start a new conversation"
            >
              <Plus size={16} aria-hidden="true" />
              <span>New chat</span>
            </button>
          )}
        </header>

        <section className="chat-content" aria-label="Conversation">
          {turns.length === 0 ? (
            <div className="welcome-state">
              <h1>
                {hasSelectedSources
                  ? "Your sources are ready."
                  : mode === "general"
                    ? "What can I help with today?"
                    : "Start with your own sources."}
              </h1>
              <p>
                {hasSelectedSources
                  ? "Ask a question, create notes, or search your selected sources."
                  : documents.length > 0
                    ? "Select the documents you want to use, or add a new PDF."
                    : "Add your resources for insight generation"}
              </p>
            </div>
          ) : (
            <div className="conversation-list" aria-live="polite">
              {turns.map((turn) => (
                <article className="conversation-turn" key={turn.id}>
                    <span className="turn-mode">
                      {turn.mode === "general"
                        ? "General chat"
                        : turn.mode === "ask"
                          ? "Answer"
                          : turn.mode === "notes"
                            ? "Research notes"
                            : turn.mode === "image"
                              ? "Image analysis"
                              : "Search results"}
                    </span>
                  <div className="user-message">
                    <p>{turn.question}</p>
                  </div>

                  {turn.response && (
                    <div className="assistant-message">
                      <span className="assistant-mark" aria-hidden="true">
                        <BookOpen size={16} strokeWidth={1.8} />
                      </span>

                      <div className="assistant-content">
                        <StreamedMarkdown
                          text={turn.response.answer}
                          streaming={turn.streaming ?? false}
                        />

                        {turn.response.sources.length > 0 && (
                          <details className="sources-section">
                            <summary className="sources-summary">
                              <span>Sources ({turn.response.sources.length})</span>
                              <ChevronDown size={14} aria-hidden="true" />
                            </summary>
                            <div className="source-list">
                              {turn.response.sources.map((source, index) => {
                                const evidence = turn.response?.retrieval.find(
                                  (item) => item.chunk_id === source.chunk_id,
                                );

                                return (
                                  <details
                                    className="source-item"
                                    key={source.chunk_id}
                                  >
                                    <summary>
                                      <BookOpen size={14} aria-hidden="true" />
                                      <span className="source-number">
                                        [{index + 1}]
                                      </span>
                                      <span>
                                        {displayDocumentName(
                                          source.document_id,
                                          documents,
                                        )}
                                        , {source.page_label}
                                      </span>
                                      <ArrowUpRight
                                        className="source-open-icon"
                                        size={14}
                                        aria-hidden="true"
                                      />
                                    </summary>
                                    {evidence && (
                                      <p className="source-excerpt">
                                        {evidence.text}
                                      </p>
                                    )}
                                  </details>
                                );
                              })}
                            </div>
                          </details>
                        )}
                      </div>
                    </div>
                  )}

                  {turn.results && (
                    <div className="assistant-message">
                      <span className="assistant-mark" aria-hidden="true">
                        <Search size={16} strokeWidth={1.8} />
                      </span>
                      <div className="assistant-content search-results">
                        {turn.results.length === 0 ? (
                          <p className="answer-copy">
                            No matching passages were found.
                          </p>
                        ) : (
                          turn.results.map((result, index) => (
                            <details
                              className="source-item"
                              key={result.chunk_id}
                            >
                              <summary>
                                <span className="source-number">[{index + 1}]</span>
                                <span>
                                  {displayDocumentName(
                                    result.document_id,
                                    documents,
                                  )}
                                  , {citationWithoutDocumentId(result)}
                                </span>
                                <ArrowUpRight
                                  className="source-open-icon"
                                  size={14}
                                  aria-hidden="true"
                                />
                              </summary>
                              <p className="source-excerpt">{result.text}</p>
                            </details>
                          ))
                        )}
                      </div>
                    </div>
                  )}

                  {turn.imageAnalysis && (
                    <div className="assistant-message">
                      <span className="assistant-mark" aria-hidden="true">
                        <ImageIcon size={16} strokeWidth={1.8} />
                      </span>
                      <div className="assistant-content">
                        <StreamedMarkdown
                          text={turn.imageAnalysis.analysis}
                          streaming={turn.streaming ?? false}
                        />
                        <p className="image-analysis-notice">
                          {turn.imageAnalysis.notice}
                        </p>
                      </div>
                    </div>
                  )}

                  {turn.error && (
                    <p className="answer-error" role="alert">
                      {turn.error}
                    </p>
                  )}
                </article>
              ))}

              {loading && !hasStreamedText && (
                <div className="assistant-message thinking" role="status">
                  <span className="assistant-mark" aria-hidden="true">
                    <BookOpen size={16} strokeWidth={1.8} />
                  </span>
                  <span>{processingMessage}</span>
                  <span className="thinking-dots" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </span>
                </div>
              )}
              <div ref={endOfConversation} />
            </div>
          )}
        </section>

        <footer className="composer-area">
          {documentError && (
            <p className="document-error" role="alert">
              {documentError}
            </p>
          )}

          <form
            className="composer"
            onSubmit={(event) => void askQuestion(event)}
          >
            {selectedImage && (
              <div className="image-attachment">
                <ImageIcon size={14} aria-hidden="true" />
                <span>{selectedImage.name}</span>
                <button
                  type="button"
                  onClick={() => setSelectedImage(null)}
                  aria-label="Remove image attachment"
                  title="Remove image"
                >
                  <X size={14} aria-hidden="true" />
                </button>
              </div>
            )}
            {!selectedImage && (
              <div
                className="composer-modes"
                role="group"
                aria-label="Response mode"
              >
                <button
                  type="button"
                  className={mode === "general" ? "composer-mode-active" : ""}
                  aria-pressed={mode === "general"}
                  onClick={() => setMode("general")}
                >
                  <MessageSquare size={14} aria-hidden="true" />
                  Chat
                </button>
                {hasSelectedSources && (
                  <>
                <button
                  type="button"
                  className={mode === "ask" ? "composer-mode-active" : ""}
                  aria-pressed={mode === "ask"}
                  onClick={() => setMode("ask")}
                >
                  <BookOpen size={14} aria-hidden="true" />
                  Ask
                </button>
                <button
                  type="button"
                  className={mode === "notes" ? "composer-mode-active" : ""}
                  aria-pressed={mode === "notes"}
                  onClick={() => setMode("notes")}
                >
                  <FileText size={14} aria-hidden="true" />
                  Notes
                </button>
                <button
                  type="button"
                  className={mode === "search" ? "composer-mode-active" : ""}
                  aria-pressed={mode === "search"}
                  onClick={() => setMode("search")}
                >
                  <Search size={14} aria-hidden="true" />
                  Search
                </button>
                  </>
                )}
              </div>
            )}
            <label className="composer-question-label" htmlFor="question-input">
              {selectedImage
                ? "Ask about this image"
                : mode === "general"
                  ? "Message"
                  : mode === "ask"
                  ? "Ask your sources"
                  : mode === "notes"
                    ? "Create notes for a specific need"
                    : "Search selected sources"}
            </label>
            <textarea
              ref={queryTextareaRef}
              id="question-input"
              value={query}
              onChange={(event) => {
                setQuery(event.currentTarget.value);
                event.currentTarget.style.height = "auto";
                event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 180)}px`;
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  void askQuestion();
                }
              }}
              placeholder={
                selectedImage
                  ? "What should I look for in this image?"
                  : mode === "general"
                    ? "Ask anything..."
                  : !hasSelectedSources
                    ? "Select sources below or add a PDF to begin..."
                    : mode === "ask"
                      ? "Ask about your selected documents..."
                      : mode === "notes"
                        ? "What do you need to understand or decide?"
                        : "Search your selected documents..."
              }
              rows={1}
            />
            <div className="composer-controls">
              <div className="document-toolbar">
                <label className={`upload-button${uploading ? " upload-button-disabled" : ""}`}>
                  <Upload size={15} aria-hidden="true" />
                  <span>{uploading ? "Indexing PDF..." : "Add PDF"}</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf,.pdf"
                    multiple
                    disabled={uploading}
                    onChange={(event) => void uploadFiles(event)}
                    aria-label="Upload PDF documents"
                  />
                </label>

                <label className="upload-button image-upload-button">
                  <ImageIcon size={15} aria-hidden="true" />
                  <span>Analyze image</span>
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/gif,image/webp"
                    onChange={chooseImage}
                    aria-label="Choose an image to analyze"
                  />
                </label>

                {documents.length > 0 && (
                  <details className="documents-picker">
                    <summary>
                      <span>
                        {selectedDocumentIds.length === 0
                          ? "Choose sources"
                          : `${selectedDocumentIds.length} selected`}
                      </span>
                      <ChevronDown size={14} aria-hidden="true" />
                    </summary>
                    <div className="documents-menu">
                      <button
                        className="all-documents-option"
                        type="button"
                        onClick={() =>
                          setSelectedDocumentIds(
                            documents.map((document) => document.document_id),
                          )
                        }
                      >
                        Select all uploaded documents
                      </button>
                      <button
                        className="all-documents-option"
                        type="button"
                        onClick={() => setSelectedDocumentIds([])}
                      >
                        Clear selection
                      </button>
                      <p className="documents-group-label">Your documents</p>
                      {documents.map((document) => (
                        <div className="document-option" key={document.document_id}>
                          <label>
                            <input
                              type="checkbox"
                              checked={selectedDocumentIds.includes(document.document_id)}
                              onChange={(event) =>
                                toggleDocument(document.document_id, event.target.checked)
                              }
                            />
                            <span className="document-option-name">{document.filename}</span>
                            <span className="document-option-meta">
                              {document.page_count} pages, {document.chunk_count} chunks
                            </span>
                          </label>
                          <button
                            className="delete-document-button"
                            type="button"
                            title={`Remove ${document.filename}`}
                            aria-label={`Remove ${document.filename}`}
                            onClick={() => void deleteDocument(document)}
                          >
                            <Trash2 size={14} aria-hidden="true" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
              <button
                className="send-button"
                type="submit"
                disabled={
                  loading ||
                  (!query.trim() && !selectedImage) ||
                  (mode !== "general" && !selectedImage && !hasSelectedSources)
                }
                aria-label="Send question"
                title="Send question"
              >
                <Send size={17} aria-hidden="true" />
              </button>
            </div>
          </form>
        </footer>
      </div>
    </main>
  );
}