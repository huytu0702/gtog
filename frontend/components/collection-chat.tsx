'use client';

import React from 'react';
import { useMutation } from '@tanstack/react-query';
import { searchApi, Collection, SearchResult, ConversationTurn } from '@/lib/api';
import { NBButton } from '@/components/ui/NBButton';
import { NBCard } from '@/components/ui/NBCard';
import { NBInput } from '@/components/ui/NBInput';
import { Send, Bot, User, Settings, Loader2, Globe, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

const DATASET_COLORS: Record<string, string> = {
    Reports:       'bg-blue-100 text-blue-800 border-blue-400',
    Entities:      'bg-green-100 text-green-800 border-green-400',
    Relationships: 'bg-purple-100 text-purple-800 border-purple-400',
    Sources:       'bg-orange-100 text-orange-800 border-orange-400',
    Claims:        'bg-red-100 text-red-800 border-red-400',
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
        const entry = lookup[id];
        if (!entry) return `#${id}`;
        const desc = entry.description ? ` — ${entry.description.slice(0, 120)}` : '';
        return `#${id}: ${entry.name}${desc}`;
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
                'inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-bold border rounded-sm mx-0.5 align-middle cursor-help',
                color
            )}
            title={tooltip}
        >
            {dataset} <span className="opacity-70">({ids})</span>
        </span>
    );
}

// Regex: matches [Data: Dataset1 (ids); Dataset2 (ids)]
const CITATION_RE = /\[Data:\s*((?:[^[\]]+?))\]/g;
const ENTRY_RE = /([A-Za-z]+)\s*\(([^)]+)\)/g;

function MessageContent({ text, context }: { text: string; context: ContextLookup | null }) {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    CITATION_RE.lastIndex = 0;
    while ((match = CITATION_RE.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
        }
        const inner = match[1];
        const badges: React.ReactNode[] = [];
        let entryMatch: RegExpExecArray | null;
        ENTRY_RE.lastIndex = 0;
        while ((entryMatch = ENTRY_RE.exec(inner)) !== null) {
            badges.push(
                <CitationBadge key={badges.length} dataset={entryMatch[1]} ids={entryMatch[2]} context={context} />
            );
        }
        parts.push(<span key={match.index}>{badges}</span>);
        lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }

    return <p className="whitespace-pre-wrap font-medium leading-relaxed">{parts}</p>;
}

interface CollectionChatProps {
    collection: Collection;
}

type Message = {
    role: 'user' | 'bot';
    content: string;
    context?: any;
    method?: string;
};

type SearchMethod = 'global' | 'local' | 'tog' | 'drift' | 'agent' | 'web';

// Summarize when history exceeds this many user turns
const SUMMARIZE_THRESHOLD = 6;

export function CollectionChat({ collection }: CollectionChatProps) {
    const [messages, setMessages] = React.useState<Message[]>([
        { role: 'bot', content: `Hello! I'm ready to answer questions about "${collection.name}".` },
    ]);
    const [input, setInput] = React.useState('');
    const [method, setMethod] = React.useState<SearchMethod>('agent');
    const [isStreaming, setIsStreaming] = React.useState(false);
    const [convHistory, setConvHistory] = React.useState<ConversationTurn[]>([]);
    const [convSummary, setConvSummary] = React.useState<string | undefined>(undefined);
    const scrollRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const searchMutation = useMutation({
        mutationFn: async (query: string) => {
            switch (method) {
                case 'global': return searchApi.global(collection.id, query);
                case 'local': return searchApi.local(collection.id, query);
                case 'tog': return searchApi.tog(collection.id, query);
                case 'drift': return searchApi.drift(collection.id, query);
                case 'agent': return searchApi.agent(collection.id, query, convHistory, convSummary);
                case 'web': return searchApi.web(collection.id, query);
                default: return searchApi.agent(collection.id, query, convHistory, convSummary);
            }
        },
        onSuccess: async (data: SearchResult | any, query: string) => {
            const methodUsed = data.method_used || data.method;
            const reasoning = data.router_reasoning;
            let content = data.response;

            // Add reasoning for agent search
            if (reasoning) {
                content = `[${methodUsed.toUpperCase()} search selected: ${reasoning}]\n\n${content}`;
            }

            setMessages((prev) => [
                ...prev,
                { role: 'bot', content, context: data.context_data, method: methodUsed },
            ]);

            // Update conversation history for agent method
            if (method === 'agent') {
                const userTurn: ConversationTurn = {
                    role: 'user',
                    content: query,
                    rewritten_query: data.rewritten_query,
                    method_used: methodUsed,
                };
                const assistantTurn: ConversationTurn = {
                    role: 'assistant',
                    content: data.response,
                };
                const newHistory = [...convHistory, userTurn, assistantTurn];

                // Count user turns
                const userTurnCount = newHistory.filter(t => t.role === 'user').length;
                if (userTurnCount >= SUMMARIZE_THRESHOLD) {
                    try {
                        const result = await searchApi.summarize(collection.id, newHistory, convSummary);
                        setConvSummary(result.summary);
                        setConvHistory(result.trimmed_history);
                    } catch {
                        // On summarization failure, keep history as-is
                        setConvHistory(newHistory);
                    }
                } else {
                    setConvHistory(newHistory);
                }
            }
        },
        onError: (error: Error) => {
            setMessages((prev) => [
                ...prev,
                { role: 'bot', content: `Error: ${error.message}` },
            ]);
        },
    });

    const handleSend = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || searchMutation.isPending) return;

        const query = input;
        setInput('');
        setMessages((prev) => [...prev, { role: 'user', content: query }]);
        searchMutation.mutate(query);
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 lg:h-[600px]">
            {/* Chat Area */}
            <div className="lg:col-span-3 flex flex-col h-[600px] lg:h-full min-h-0">
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
                                            'p-4 border-2 border-black shadow-hard-sm',
                                            msg.role === 'user' ? 'bg-white' : 'bg-white'
                                        )}
                                    >
                                        <MessageContent text={msg.content} context={msg.context ?? null} />
                                    </div>

                        {msg.method && (
                            <div className="text-xs text-gray-500 font-bold flex items-center gap-1">
                                {msg.method === 'web' && <Globe className="w-3 h-3" />}
                                {msg.method === 'agent' && <Sparkles className="w-3 h-3" />}
                                Used {msg.method} search
                            </div>
                        )}
                                </div>
                            </div>
                        ))}

                        {searchMutation.isPending && (
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
                                disabled={searchMutation.isPending}
                            />
                            <NBButton type="submit" disabled={searchMutation.isPending || !input.trim()}>
                                <Send className="w-5 h-5" />
                            </NBButton>
                        </form>
                    </div>
                </NBCard>
            </div>

            {/* Settings Sidebar */}
            <div className="lg:col-span-1">
                <NBCard className="h-full bg-white">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                        <Settings className="w-5 h-5" />
                        Search Settings
                    </h3>

                    <div className="space-y-4">
                        <div>
                            <label className="block font-bold mb-2">Search Method</label>
                            <div className="space-y-2">
                                {([
                                    { id: 'agent', label: 'Auto (Agent)', icon: Sparkles },
                                    { id: 'global', label: 'Global', icon: null },
                                    { id: 'local', label: 'Local', icon: null },
                                    { id: 'tog', label: 'ToG', icon: null },
                                    { id: 'drift', label: 'DRIFT', icon: null },
                                    { id: 'web', label: 'Web Search', icon: Globe },
                                ] as const).map((m) => (
                                    <button
                                        key={m.id}
                                        onClick={() => setMethod(m.id)}
                                        className={cn(
                                            'w-full text-left px-4 py-3 border-2 border-black font-bold transition-all uppercase flex items-center gap-2',
                                            method === m.id
                                                ? 'bg-main shadow-hard-sm translate-x-[-2px] translate-y-[-2px]'
                                                : 'bg-white hover:bg-gray-100'
                                        )}
                                    >
                                        {m.icon && <m.icon className="w-4 h-4" />}
                                        {m.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="p-4 bg-yellow-100 border-2 border-black text-sm">
                            <p className="font-bold mb-1">Tip:</p>
                            {method === 'agent' && 'Automatically selects the best search method for your query.'}
                            {method === 'global' && 'Best for overview questions about the entire collection.'}
                            {method === 'local' && 'Best for specific questions about entities and their relationships.'}
                            {method === 'tog' && 'Think-on-Graph: Good for complex multi-hop reasoning.'}
                            {method === 'drift' && 'DRIFT: Dynamic reasoning for hypothetical scenarios.'}
                            {method === 'web' && 'Search the web for current information not in your collection.'}
                        </div>
                    </div>
                </NBCard>
            </div>
        </div>
    );
}
