import axios from 'axios';

function normalizeBaseUrl(value?: string): string {
    if (!value) return '';
    return value.endsWith('/') ? value.slice(0, -1) : value;
}

export class ApiStatusError extends Error {
    status: number;

    constructor(status: number) {
        super(status === 401 || status === 403 ? 'Authentication required' : `Request failed with status ${status}`);
        this.name = 'ApiStatusError';
        this.status = status;
    }
}

const API_HOST_BASE_URL = normalizeBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL) || 'http://127.0.0.1:8000';
const API_BASE_URL = `${API_HOST_BASE_URL}/api`;

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    validateStatus: () => true,
});

api.interceptors.response.use((response) => {
    if (response.status >= 400) {
        throw new ApiStatusError(response.status);
    }
    return response;
});

export interface Collection {
    id: string;
    name: string;
    description: string | null;
    created_at: string;
    document_count: number;
    indexed: boolean;
}

export interface Document {
    name: string;
    size: number;
    uploaded_at: string;
}

export interface IndexingStatus {
    collection_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    message: string;
    started_at: string | null;
    completed_at: string | null;
    error: string | null;
}

export interface SearchResult {
    query: string;
    response: string;
    context_data: Record<string, unknown> | null;
    method: string;
}

export interface ConversationTurn {
    role: 'user' | 'assistant';
    content: string;
    rewritten_query?: string;
    method_used?: string;
}

export interface AgentSearchResult {
    method_used: string;
    router_reasoning: string;
    rewritten_query?: string;
    response: string;
    sources: Array<{
        id: number;
        title: string;
        url?: string;
        text_unit_id?: string;
    }>;
    web_response?: string | null;
    web_sources?: Array<{
        id: number;
        title: string;
        url?: string;
    }>;
}

export interface SummarizeResult {
    summary: string;
    trimmed_history: ConversationTurn[];
}

export interface WebSearchResult {
    query: string;
    response: string;
    sources: Array<{
        id: number;
        title: string;
        url?: string;
    }>;
    method: string;
}

export const collectionsApi = {
    list: async () => {
        const response = await api.get<{ collections: Collection[]; total: number }>('/collections');
        return response.data;
    },
    create: async (data: { name: string; description?: string }) => {
        const response = await api.post<Collection>('/collections', data);
        return response.data;
    },
    get: async (id: string) => {
        const response = await api.get<Collection>(`/collections/${id}`);
        return response.data;
    },
    delete: async (id: string) => {
        await api.delete(`/collections/${id}`);
    },
};

export const documentsApi = {
    list: async (collectionId: string) => {
        const response = await api.get<{ documents: Document[]; total: number }>(`/collections/${collectionId}/documents`);
        return response.data;
    },
    upload: async (collectionId: string, file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await api.post<Document>(`/collections/${collectionId}/documents`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },
    delete: async (collectionId: string, documentName: string) => {
        await api.delete(`/collections/${collectionId}/documents/${documentName}`);
    },
};

export const indexingApi = {
    start: async (collectionId: string) => {
        const response = await api.post<IndexingStatus>(`/collections/${collectionId}/index`);
        return response.data;
    },
    getStatus: async (collectionId: string) => {
        const response = await api.get<IndexingStatus>(`/collections/${collectionId}/index`);
        return response.data;
    },
};

export const searchApi = {
    global: async (collectionId: string, query: string) => {
        const response = await api.post<SearchResult>(`/collections/${collectionId}/search/global`, {
            query,
            response_type: 'Multiple Paragraphs',
        });
        return response.data;
    },
    local: async (collectionId: string, query: string) => {
        const response = await api.post<SearchResult>(`/collections/${collectionId}/search/local`, {
            query,
            community_level: 2,
            response_type: 'Multiple Paragraphs',
        });
        return response.data;
    },
    tog: async (collectionId: string, query: string) => {
        const response = await api.post<SearchResult>(`/collections/${collectionId}/search/tog`, {
            query,
        });
        return response.data;
    },
    drift: async (collectionId: string, query: string) => {
        const response = await api.post<SearchResult>(`/collections/${collectionId}/search/drift`, {
            query,
            community_level: 2,
            response_type: 'Multiple Paragraphs',
        });
        return response.data;
    },
    agent: async (
        collectionId: string,
        query: string,
        conversationHistory: ConversationTurn[] = [],
        conversationSummary?: string,
        webSearchEnabled: boolean = false,
    ) => {
        const response = await api.post<AgentSearchResult>(`/collections/${collectionId}/search/agent`, {
            query,
            stream: false,
            conversation_history: conversationHistory,
            conversation_summary: conversationSummary,
            web_search_enabled: webSearchEnabled,
        });
        return response.data;
    },
    summarize: async (
        collectionId: string,
        conversationHistory: ConversationTurn[],
        existingSummary?: string,
    ) => {
        const response = await api.post<SummarizeResult>(`/collections/${collectionId}/search/agent/summarize`, {
            conversation_history: conversationHistory,
            existing_summary: existingSummary,
        });
        return response.data;
    },
    web: async (collectionId: string, query: string) => {
        const response = await api.post<WebSearchResult>(`/collections/${collectionId}/search/web`, {
            query,
            stream: false,
        });
        return response.data;
    },
    agentStream: (
        collectionId: string,
        query: string,
        onMessage: (data: unknown) => void,
        sessionId?: string,
    ) => {
        const params = new URLSearchParams({ query });
        if (sessionId) {
            params.set('session_id', sessionId);
        }
        const eventSource = new EventSource(
            `${API_BASE_URL}/collections/${collectionId}/search/agent/stream?${params.toString()}`,
        );

        const handleEvent = (event: MessageEvent) => {
            onMessage(JSON.parse(event.data));
        };
        eventSource.onmessage = handleEvent;
        eventSource.addEventListener('status', handleEvent as EventListener);
        eventSource.addEventListener('content', handleEvent as EventListener);
        eventSource.addEventListener('done', handleEvent as EventListener);
        eventSource.addEventListener('error', handleEvent as EventListener);

        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            eventSource.close();
        };

        return eventSource;
    },
};
