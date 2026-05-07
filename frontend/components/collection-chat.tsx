'use client';

import React from 'react';
import { useMutation } from '@tanstack/react-query';
import {
    searchApi,
    Collection,
    SearchResult,
    ConversationTurn,
    AgentSearchResult,
    AgentStreamStatusEvent,
} from '@/lib/api';
import { NBButton } from '@/components/ui/NBButton';
import { NBCard } from '@/components/ui/NBCard';
import { NBInput } from '@/components/ui/NBInput';
import { Send, Bot, User, Settings, Loader2, Globe, Sparkles, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';

const DATASET_COLORS: Record<string, string> = {
    Reports: 'bg-blue-100 text-blue-800 border-blue-400',
    Entities: 'bg-green-100 text-green-800 border-green-400',
    Relationships: 'bg-purple-100 text-purple-800 border-purple-400',
    Sources: 'bg-orange-100 text-orange-800 border-orange-400',
    Claims: 'bg-red-100 text-red-800 border-red-400',
};

// context_data shape: { [dataset: string]: { [id: string]: { name: string, description: string } } }
type ContextLookup = Record<string, Record<string, { name: string; description: string }>>;

function buildTooltip(dataset: string, ids: string, context: ContextLookup | null): string {
    if (!context) return `${dataset}: ${ids}`;
    // The backend keys are title-cased ("Entities"), match case-insensitively
    const datasetKey = Object.keys(context).find(
        (k) => k.toLowerCase() === dataset.toLowerCase()
    );
    if (!datasetKey) return `${dataset}: ${ids}`;
    const lookup = context[datasetKey];
    const lines = ids.split(',').map((rawId) => {
        const id = rawId.trim().replace('+more', '').trim();
        // Match case-insensitively since entity names may differ in case
        const entryKey = Object.keys(lookup).find((k) => k.toLowerCase() === id.toLowerCase()) ?? id;
        const entry = lookup[entryKey];
        if (!entry) return id;
        const desc = entry.description ? ` — ${entry.description.slice(0, 300)}` : '';
        return `${entry.name}${desc}`;
    });
    return lines.join('\n');
}

function CitationBadge({
    dataset,
    ids,
    context,
}: {
    dataset: string;
    ids: string;
    context: ContextLookup | null;
}) {
    const color = DATASET_COLORS[dataset] ?? 'bg-gray-100 text-gray-700 border-gray-400';
    const tooltip = buildTooltip(dataset, ids, context);
    return (
        <span
            className={cn(
                'group relative inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-bold border rounded-sm mx-0.5 align-middle cursor-help',
                color
            )}
        >
            {dataset} <span className="opacity-70">({ids})</span>
            <span className="invisible absolute bottom-full left-0 z-50 block max-h-64 w-80 max-w-[80vw] overflow-y-auto whitespace-pre-wrap border-2 border-black bg-white p-3 text-left text-xs font-medium normal-case text-black opacity-0 shadow-hard transition-opacity group-hover:visible group-hover:opacity-100">
                {tooltip}
            </span>
        </span>
    );
}

// Regex: matches [Data: Dataset1 (ids); Dataset2 (ids)]
const CITATION_RE = /\[Data:\s*((?:[^[\]]+?))\]/g;
const ENTRY_RE = /([A-Za-z]+)\s*\(([^)]+)\)/g;

const MARKDOWN_COMPONENTS = {
    p: ({ children }: { children?: React.ReactNode }) => <span className="font-medium leading-relaxed">{children}</span>,
    ul: ({ children }: { children?: React.ReactNode }) => <ul className="mb-2 list-disc pl-5">{children}</ul>,
    ol: ({ children }: { children?: React.ReactNode }) => <ol className="mb-2 list-decimal pl-5">{children}</ol>,
    li: ({ children }: { children?: React.ReactNode }) => <li className="mb-1">{children}</li>,
    h1: ({ children }: { children?: React.ReactNode }) => <h1 className="mb-2 block text-xl font-bold">{children}</h1>,
    h2: ({ children }: { children?: React.ReactNode }) => <h2 className="mb-2 block text-lg font-bold">{children}</h2>,
    h3: ({ children }: { children?: React.ReactNode }) => <h3 className="mb-2 block text-base font-bold">{children}</h3>,
    code: ({ children }: { children?: React.ReactNode }) => <code className="rounded bg-gray-100 px-1 py-0.5 text-sm">{children}</code>,
    pre: ({ children }: { children?: React.ReactNode }) => <pre className="mb-2 overflow-x-auto rounded border-2 border-black bg-gray-50 p-3">{children}</pre>,
    a: ({ children, href }: { children?: React.ReactNode; href?: string }) => (
        <a href={href} target="_blank" rel="noreferrer" className="text-blue-600 underline hover:text-blue-800">
            {children}
        </a>
    ),
    blockquote: ({ children }: { children?: React.ReactNode }) => <blockquote className="mb-2 border-l-4 border-black pl-3 italic">{children}</blockquote>,
};

function MessageContent({ text, context }: { text: string; context: ContextLookup | null }) {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    const citationRe = new RegExp(CITATION_RE.source, 'g');
    while ((match = citationRe.exec(text)) !== null) {
        if (match.index > lastIndex) {
            const markdownChunk = text.slice(lastIndex, match.index);
            parts.push(
                <ReactMarkdown
                    key={`md-${match.index}`}
                    remarkPlugins={[remarkGfm]}
                    components={MARKDOWN_COMPONENTS}
                >
                    {markdownChunk}
                </ReactMarkdown>
            );
        }

        const inner = match[1];
        const badges: React.ReactNode[] = [];
        const entryRe = new RegExp(ENTRY_RE.source, 'g');
        let entryMatch: RegExpExecArray | null;
        while ((entryMatch = entryRe.exec(inner)) !== null) {
            badges.push(
                <CitationBadge key={badges.length} dataset={entryMatch[1]} ids={entryMatch[2]} context={context} />
            );
        }
        parts.push(<span key={`citation-${match.index}`}>{badges}</span>);
        lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
        const markdownChunk = text.slice(lastIndex);
        parts.push(
            <ReactMarkdown
                key="md-tail"
                remarkPlugins={[remarkGfm]}
                components={MARKDOWN_COMPONENTS}
            >
                {markdownChunk}
            </ReactMarkdown>
        );
    }

    return <div className="whitespace-pre-wrap">{parts}</div>;
}

function MethodBadge({ method, webSearchTriggered }: { method: string; webSearchTriggered?: boolean }) {
    return (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs font-bold uppercase">
            <span className="inline-flex items-center gap-1 border-2 border-black bg-main px-2 py-1 shadow-hard-sm">
                {method === 'agent' && <Sparkles className="w-3 h-3" />}
                {method} search
            </span>
            {webSearchTriggered && (
                <span className="inline-flex items-center gap-1 border-2 border-blue-400 bg-blue-100 px-2 py-1 text-blue-700 shadow-hard-sm">
                    <Globe className="w-3 h-3" />
                    Web-augmented
                </span>
            )}
        </div>
    );
}

function ProcessingStatus({ steps, currentStep, streamError }: {
    steps: StatusStep[];
    currentStep?: string;
    streamError?: string;
}) {
    return (
        <div className="space-y-3">
            <div className="flex items-center gap-2 font-bold">
                <Loader2 className="w-4 h-4 animate-spin" />
                Thinking...
            </div>
            <div className="space-y-1 text-sm">
                {steps.map((step, stepIdx) => (
                    <div key={`${step.step}-${stepIdx}`} className="flex items-center gap-2">
                        {currentStep === step.step ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                            <span className="w-3 h-3 inline-flex items-center justify-center">•</span>
                        )}
                        <span>{step.message}</span>
                    </div>
                ))}
            </div>
            {streamError && (
                <div className="text-sm text-red-700 font-bold">Error: {streamError}</div>
            )}
        </div>
    );
}

interface CollectionChatProps {
    collection: Collection;
}

type StatusStep = {
    step: string;
    message: string;
    method?: string;
};

type Message = {
    role: 'user' | 'bot';
    content: string;
    context?: Record<string, unknown> | null;
    method?: string;
    webContent?: string;
    webSources?: Array<{ id: number; title: string; url?: string }>;
    webSearchTriggered?: boolean;
    statusSteps?: StatusStep[];
    currentStep?: string;
    streamError?: string;
};

type SearchMethod = 'global' | 'local' | 'tog' | 'drift' | 'agent';

// Summarize when history exceeds this many user turns
const SUMMARIZE_THRESHOLD = 6;

const STREAMING_PLACEHOLDER = 'Preparing response...';

function formatStatusMessage(event: AgentStreamStatusEvent): string {
    switch (event.step) {
        case 'routing':
            return 'Analyzing question...';
        case 'routed':
            return event.method ? `Selected ${event.method.toUpperCase()} search...` : 'Selected search method...';
        case 'searching':
            return 'Searching knowledge graph...';
        case 'judging_sufficiency':
            return 'Checking information sufficiency...';
        case 'web_searching':
            return 'Searching the web for additional context...';
        default:
            return event.message || 'Processing...';
    }
}

export function CollectionChat({ collection }: CollectionChatProps) {
    const [messages, setMessages] = React.useState<Message[]>([
        { role: 'bot', content: `Hello! I'm ready to answer questions about "${collection.name}".` },
    ]);
    const [input, setInput] = React.useState('');
    const [method, setMethod] = React.useState<SearchMethod>('agent');
    const [showAdvancedMethods, setShowAdvancedMethods] = React.useState(false);

    // Auto-expand manual methods panel when an advanced method is active
    React.useEffect(() => {
        if (method !== 'agent') setShowAdvancedMethods(true);
    }, [method]);
    const [isStreaming, setIsStreaming] = React.useState(false);
    const [convHistory, setConvHistory] = React.useState<ConversationTurn[]>([]);
    const [convSummary, setConvSummary] = React.useState<string | undefined>(undefined);
    const scrollRef = React.useRef<HTMLDivElement>(null);
    const streamAbortRef = React.useRef<AbortController | null>(null);
    const streamingMessageIndexRef = React.useRef<number | null>(null);

    React.useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    React.useEffect(() => () => {
        streamAbortRef.current?.abort();
    }, []);

    const updateConversation = React.useCallback(async (query: string, methodUsed: string, response: string, rewrittenQuery?: string) => {
        const userTurn: ConversationTurn = {
            role: 'user',
            content: query,
            rewritten_query: rewrittenQuery,
            method_used: methodUsed,
        };
        const assistantTurn: ConversationTurn = {
            role: 'assistant',
            content: response,
        };
        const newHistory = [...convHistory, userTurn, assistantTurn];
        const userTurnCount = newHistory.filter((t) => t.role === 'user').length;

        if (userTurnCount >= SUMMARIZE_THRESHOLD) {
            try {
                const result = await searchApi.summarize(collection.id, newHistory, convSummary);
                setConvSummary(result.summary);
                setConvHistory(result.trimmed_history);
                return;
            } catch {
                setConvHistory(newHistory);
                return;
            }
        }

        setConvHistory(newHistory);
    }, [collection.id, convHistory, convSummary]);

    const searchMutation = useMutation({
        mutationFn: async (query: string) => {
            // Helper to get the base search result for the selected method
            const fetchBase = () => {
                switch (method) {
                    case 'global': return searchApi.global(collection.id, query);
                    case 'local': return searchApi.local(collection.id, query);
                    case 'tog': return searchApi.tog(collection.id, query);
                    case 'drift': return searchApi.drift(collection.id, query);
                    case 'agent': return searchApi.agent(collection.id, query, convHistory, convSummary);
                    default: return searchApi.agent(collection.id, query, convHistory, convSummary);
                }
            };

            return fetchBase();
        },
        onSuccess: async (data: SearchResult | AgentSearchResult, query: string) => {
            const methodUsed = 'method_used' in data ? data.method_used : data.method;
            const reasoning = 'router_reasoning' in data ? data.router_reasoning : undefined;
            const content = data.response;

            setMessages((prev) => [
                ...prev,
                {
                    role: 'bot',
                    content,
                    context: 'context_data' in data ? data.context_data : null,
                    method: methodUsed,
                    webContent: 'web_response' in data ? data.web_response ?? undefined : undefined,
                    webSources: 'web_sources' in data ? data.web_sources ?? undefined : undefined,
                    webSearchTriggered: 'web_search_triggered' in data ? data.web_search_triggered ?? false : false,
                    statusSteps: reasoning
                        ? [{ step: 'routed', message: `[${methodUsed.toUpperCase()} search selected: ${reasoning}]`, method: methodUsed }]
                        : undefined,
                },
            ]);

            if (method === 'agent') {
                await updateConversation(
                    query,
                    methodUsed,
                    data.response,
                    'rewritten_query' in data ? data.rewritten_query : undefined,
                );
            }
        },
        onError: (error: Error) => {
            setMessages((prev) => [
                ...prev,
                { role: 'bot', content: `Error: ${error.message}` },
            ]);
        },
    });

    const updateStreamingMessage = React.useCallback((updater: (message: Message) => Message) => {
        const index = streamingMessageIndexRef.current;
        if (index === null) return;

        setMessages((prev) => {
            if (!prev[index] || prev[index].role !== 'bot') return prev;
            const updated = updater(prev[index]);
            const next = [...prev];
            next[index] = updated;
            return next;
        });
    }, []);

    const handleAgentStream = React.useCallback(async (query: string) => {
        setIsStreaming(true);
        const abortController = new AbortController();
        streamAbortRef.current = abortController;

        setMessages((prev) => {
            const startIndex = prev.length + 1;
            streamingMessageIndexRef.current = startIndex;
            return [
                ...prev,
                { role: 'user', content: query },
                {
                    role: 'bot',
                    content: '',
                    statusSteps: [{ step: 'routing', message: 'Analyzing question...' }],
                    currentStep: 'routing',
                },
            ];
        });

        let finalMethodUsed = 'agent';
        let finalRewrittenQuery: string | undefined;
        let finalResponse = '';

        try {
            await searchApi.agentStreamPost(
                collection.id,
                query,
                {
                    onStatus: (event) => {
                        const mappedMessage = formatStatusMessage(event);
                        updateStreamingMessage((message) => {
                            const prevSteps = message.statusSteps ?? [];
                            const exists = prevSteps.some((step) => step.step === event.step);
                            const nextSteps = exists
                                ? prevSteps.map((step) => (
                                    step.step === event.step
                                        ? { step: event.step, message: mappedMessage, method: event.method }
                                        : step
                                ))
                                : [...prevSteps, { step: event.step, message: mappedMessage, method: event.method }];

                            const content = message.content === STREAMING_PLACEHOLDER ? '' : message.content;
                            return {
                                ...message,
                                content,
                                currentStep: event.step,
                                method: event.method ?? message.method,
                                statusSteps: nextSteps,
                            };
                        });
                    },
                    onContent: (event) => {
                        finalResponse += event.chunk;
                        updateStreamingMessage((message) => ({
                            ...message,
                            content: finalResponse,
                        }));
                    },
                    onDone: (event) => {
                        finalMethodUsed = event.method_used || finalMethodUsed;
                        finalRewrittenQuery = event.rewritten_query;
                        updateStreamingMessage((message) => ({
                            ...message,
                            content: finalResponse || message.content || 'No response content returned.',
                            currentStep: undefined,
                            context: event.context_data ?? null,
                            method: event.method_used,
                            webContent: event.web_response ?? undefined,
                            webSources: event.web_sources ?? undefined,
                            webSearchTriggered: event.web_search_triggered ?? false,
                            statusSteps: event.router_reasoning
                                ? [
                                    ...(message.statusSteps ?? []),
                                    {
                                        step: 'reasoning',
                                        message: `[${event.method_used.toUpperCase()} search selected: ${event.router_reasoning}]`,
                                        method: event.method_used,
                                    },
                                ]
                                : message.statusSteps,
                        }));
                    },
                    onError: (event) => {
                        streamAbortRef.current?.abort();
                        updateStreamingMessage((message) => ({
                            ...message,
                            currentStep: undefined,
                            streamError: event.error,
                            content: message.content && message.content !== STREAMING_PLACEHOLDER
                                ? message.content
                                : `Error: ${event.error}`,
                        }));
                    },
                },
                convHistory,
                convSummary,
                abortController.signal,
            );

            if (finalResponse) {
                await updateConversation(query, finalMethodUsed, finalResponse, finalRewrittenQuery);
            }
        } catch (error) {
            if ((error as Error).name !== 'AbortError') {
                updateStreamingMessage((message) => ({
                    ...message,
                    currentStep: undefined,
                    streamError: (error as Error).message,
                    content: `Error: ${(error as Error).message}`,
                }));
            }
        } finally {
            setIsStreaming(false);
            streamAbortRef.current = null;
            streamingMessageIndexRef.current = null;
        }
    }, [collection.id, convHistory, convSummary, updateConversation, updateStreamingMessage]);

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || searchMutation.isPending || isStreaming) return;

        const query = input;
        setInput('');

        if (method === 'agent') {
            await handleAgentStream(query);
            return;
        }

        setMessages((prev) => [...prev, { role: 'user', content: query }]);
        searchMutation.mutate(query);
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6" style={{ height: 'calc(100vh - 120px)' }}>
            {/* Chat Area */}
            <div className="lg:col-span-3 flex flex-col h-full min-h-0">
                <NBCard className="flex-1 h-full flex flex-col p-0 overflow-hidden bg-gray-50">
                    {/* Messages */}
                    <div
                        ref={scrollRef}
                        className="flex-1 overflow-y-auto p-6 space-y-6 [&::-webkit-scrollbar]:w-3 [&::-webkit-scrollbar-track]:bg-white [&::-webkit-scrollbar-thumb]:bg-black [&::-webkit-scrollbar-thumb]:border-2 [&::-webkit-scrollbar-thumb]:border-white"
                    >
                        {messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={cn(
                                    'flex gap-4 max-w-[80%]',
                                    msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
                                )}
                            >
                                <div
                                    className={cn(
                                        'w-10 h-10 rounded-none border-2 border-black flex items-center justify-center flex-shrink-0 shadow-hard-sm',
                                        msg.role === 'user' ? 'bg-secondary' : 'bg-main'
                                    )}
                                >
                                    {msg.role === 'user' ? <User className="w-6 h-6" /> : <Bot className="w-6 h-6" />}
                                </div>

                                <div className="space-y-2">
                                    <div
                                        className={cn(
                                            'p-4 border-2 border-black shadow-hard-sm bg-white',
                                            msg.statusSteps && msg.role === 'bot' ? 'min-w-72' : ''
                                        )}
                                    >
                                        {msg.content && (
                                            <MessageContent text={msg.content} context={(msg.context as ContextLookup | null) ?? null} />
                                        )}
                                        {msg.statusSteps && msg.statusSteps.length > 0 && (msg.currentStep || msg.streamError) && (
                                            <div className={msg.content ? 'mt-3 border-t-2 border-black pt-3' : ''}>
                                                <ProcessingStatus
                                                    steps={msg.statusSteps}
                                                    currentStep={msg.currentStep}
                                                    streamError={msg.streamError}
                                                />
                                            </div>
                                        )}
                                        {!msg.content && !msg.currentStep && !msg.streamError && (
                                            <MessageContent text={msg.content} context={(msg.context as ContextLookup | null) ?? null} />
                                        )}
                                        {msg.method && <MethodBadge method={msg.method} webSearchTriggered={msg.webSearchTriggered} />}
                                    </div>

                                    {msg.webContent && (
                                        <div className="p-4 border-2 border-black shadow-hard-sm bg-blue-50">
                                            <div className="flex items-center gap-1 mb-2 text-xs font-bold text-blue-700">
                                                <Globe className="w-3 h-3" />
                                                Web Search Results
                                            </div>
                                            <MessageContent text={msg.webContent} context={null} />
                                            {msg.webSources && msg.webSources.length > 0 && (
                                                <div className="mt-2 flex flex-wrap gap-2">
                                                    {msg.webSources.map((s) => (
                                                        <a
                                                            key={s.id}
                                                            href={s.url}
                                                            target="_blank"
                                                            rel="noreferrer"
                                                            className="text-xs text-blue-600 underline hover:text-blue-800"
                                                        >
                                                            [{s.id}] {s.title}
                                                        </a>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}

                        {searchMutation.isPending && method !== 'agent' && (
                            <div className="flex gap-4 max-w-[80%]">
                                <div className="w-10 h-10 bg-main border-2 border-black flex items-center justify-center flex-shrink-0 shadow-hard-sm">
                                    <Bot className="w-6 h-6" />
                                </div>
                                <div className="p-4 bg-white border-2 border-black shadow-hard-sm flex items-center gap-2">
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span className="font-bold">Thinking...</span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-white border-t-3 border-black">
                        <form onSubmit={handleSend} className="flex gap-4">
                            <NBInput
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask a question..."
                                className="flex-1"
                                disabled={searchMutation.isPending || isStreaming}
                            />
                            <NBButton type="submit" disabled={searchMutation.isPending || isStreaming || !input.trim()}>
                                <Send className="w-5 h-5" />
                            </NBButton>
                        </form>
                    </div>
                </NBCard>
            </div>

            {/* Settings Sidebar */}
            <div className="lg:col-span-1 h-full min-h-0">
                <NBCard className="h-full bg-white overflow-y-auto">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                        <Settings className="w-5 h-5" />
                        Search Settings
                    </h3>

                    <div className="space-y-4">
                        <div>
                            <label className="block font-bold mb-2">Search Method</label>
                            <div className="space-y-2">
                                {/* Auto (Agent) — always visible */}
                                <button
                                    onClick={() => setMethod('agent')}
                                    className={cn(
                                        'w-full text-left px-4 py-3 border-2 border-black font-bold transition-all uppercase flex items-center gap-2',
                                        method === 'agent'
                                            ? 'bg-main shadow-hard-sm translate-x-[-2px] translate-y-[-2px]'
                                            : 'bg-white hover:bg-gray-100'
                                    )}
                                >
                                    <Sparkles className="w-4 h-4" />
                                    Auto (Agent)
                                </button>

                                {/* Toggle advanced methods */}
                                <button
                                    onClick={() => setShowAdvancedMethods((v) => !v)}
                                    className="w-full text-left px-4 py-2 border-2 border-black border-dashed font-bold text-xs uppercase flex items-center justify-between text-gray-500 hover:bg-gray-50 transition-all"
                                >
                                    <span>Manual Methods</span>
                                    <ChevronDown
                                        className={cn('w-4 h-4 transition-transform duration-200', showAdvancedMethods && 'rotate-180')}
                                    />
                                </button>

                                {/* Collapsible: Global, Local, ToG, DRIFT */}
                                {showAdvancedMethods && (
                                    <div className="space-y-2">
                                        {([
                                            { id: 'global', label: 'Global' },
                                            { id: 'local', label: 'Local' },
                                            { id: 'tog', label: 'ToG' },
                                            { id: 'drift', label: 'DRIFT' },
                                        ] as const).map((m) => (
                                            <button
                                                key={m.id}
                                                onClick={() => setMethod(m.id)}
                                                className={cn(
                                                    'w-full text-left px-4 py-3 border-2 border-black font-bold transition-all uppercase',
                                                    method === m.id
                                                        ? 'bg-main shadow-hard-sm translate-x-[-2px] translate-y-[-2px]'
                                                        : 'bg-white hover:bg-gray-100'
                                                )}
                                            >
                                                {m.label}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="p-4 bg-yellow-100 border-2 border-black text-sm">
                            <p className="font-bold mb-1">Tip:</p>
                            {method === 'agent' && 'Automatically selects the best search method. Falls back to web search if the knowledge base lacks information.'}
                            {method === 'global' && 'Best for overview questions about the entire collection.'}
                            {method === 'local' && 'Best for specific questions about entities and their relationships.'}
                            {method === 'tog' && 'Think-on-Graph: Good for complex multi-hop reasoning.'}
                            {method === 'drift' && 'DRIFT: Dynamic reasoning for hypothetical scenarios.'}
                        </div>
                    </div>
                </NBCard>
            </div>
        </div>
    );
}
