import axios from 'axios';

function normalizeBaseUrl(value?: string): string {
    if (!value) return '';
    return value.endsWith('/') ? value.slice(0, -1) : value;
}

function normalizeRedirectUri(value?: string): string | null {
    const normalized = normalizeBaseUrl(value);
    return normalized || null;
}

function normalizeString(value: unknown): string | null {
    if (typeof value !== 'string') return null;
    const trimmed = value.trim();
    return trimmed || null;
}

type EasyAuthClaim = {
    typ: string;
    val: string;
};

export interface EasyAuthUser {
    isAuthenticated: boolean;
    email: string | null;
    displayName: string | null;
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
const EASY_AUTH_ME_URL = `${API_HOST_BASE_URL}/.auth/me`;
const EASY_AUTH_LOGIN_PATH = '/.auth/login/aad';

export const SIGNED_OUT_EASY_AUTH_USER: EasyAuthUser = {
    isAuthenticated: false,
    email: null,
    displayName: null,
};

function normalizeClaims(value: unknown): EasyAuthClaim[] {
    if (!Array.isArray(value)) return [];

    return value.flatMap((claim) => {
        if (!claim || typeof claim !== 'object') return [];

        const record = claim as Record<string, unknown>;
        const typ = normalizeString(record.typ) ?? normalizeString(record.type);
        const val = normalizeString(record.val) ?? normalizeString(record.value);

        return typ && val ? [{ typ, val }] : [];
    });
}

function findClaimValue(claims: EasyAuthClaim[], claimTypes: string[]): string | null {
    const normalizedClaimTypes = new Set(claimTypes.map((claimType) => claimType.toLowerCase()));

    for (const claim of claims) {
        if (normalizedClaimTypes.has(claim.typ.toLowerCase())) {
            return claim.val;
        }
    }

    return null;
}

function normalizeEasyAuthUser(payload: unknown): EasyAuthUser {
    if (!Array.isArray(payload) || payload.length === 0) {
        return SIGNED_OUT_EASY_AUTH_USER;
    }

    const entry = payload.find((candidate) => candidate && typeof candidate === 'object');
    if (!entry || typeof entry !== 'object') {
        return SIGNED_OUT_EASY_AUTH_USER;
    }

    const entryRecord = entry as Record<string, unknown>;
    const clientPrincipal =
        entryRecord.clientPrincipal && typeof entryRecord.clientPrincipal === 'object'
            ? (entryRecord.clientPrincipal as Record<string, unknown>)
            : null;
    const claims = normalizeClaims(clientPrincipal?.claims ?? entryRecord.user_claims);
    const userDetails = normalizeString(clientPrincipal?.userDetails ?? entryRecord.user_details);
    const email =
        findClaimValue(claims, [
            'email',
            'emails',
            'preferred_username',
            'upn',
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress',
        ]) ?? (userDetails?.includes('@') ? userDetails : null);
    const displayName =
        findClaimValue(claims, [
            'name',
            'preferred_username',
            'nickname',
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name',
        ]) ??
        userDetails ??
        normalizeString(entryRecord.user_id);
    const isAuthenticated = claims.length > 0 || Boolean(email) || Boolean(displayName);

    return isAuthenticated
        ? {
            isAuthenticated: true,
            email,
            displayName,
        }
        : SIGNED_OUT_EASY_AUTH_USER;
}

export async function fetchEasyAuthUser(): Promise<EasyAuthUser> {
    try {
        const response = await fetch(EASY_AUTH_ME_URL, {
            credentials: 'include',
            headers: { Accept: 'application/json' },
            cache: 'no-store',
        });

        if (!response.ok) {
            return SIGNED_OUT_EASY_AUTH_USER;
        }

        const payload: unknown = await response.json();
        return normalizeEasyAuthUser(payload);
    } catch {
        return SIGNED_OUT_EASY_AUTH_USER;
    }
}

export function getEasyAuthUserLabel(user: EasyAuthUser): string | null {
    return user.email ?? user.displayName;
}

export function buildEasyAuthLoginUrl(postLoginRedirectUri?: string): string {
    const loginUrl = new URL(`${API_HOST_BASE_URL}${EASY_AUTH_LOGIN_PATH}`);
    const redirectUri = normalizeRedirectUri(postLoginRedirectUri);
    if (redirectUri) {
        loginUrl.searchParams.set('post_login_redirect_uri', redirectUri);
    }
    return loginUrl.toString();
}

export function buildEasyAuthLogoutUrl(postLogoutRedirectUri?: string): string {
    const logoutUrl = new URL(`${API_HOST_BASE_URL}/.auth/logout`);
    const redirectUri = normalizeRedirectUri(postLogoutRedirectUri);
    if (redirectUri) {
        logoutUrl.searchParams.set('post_logout_redirect_uri', redirectUri);
    }
    return logoutUrl.toString();
}

export const EASY_AUTH_LOGIN_URL = buildEasyAuthLoginUrl();
export const EASY_AUTH_LOGOUT_URL = buildEasyAuthLogoutUrl();

export const api = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
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
    ) => {
        const response = await api.post<AgentSearchResult>(`/collections/${collectionId}/search/agent`, {
            query,
            stream: false,
            conversation_history: conversationHistory,
            conversation_summary: conversationSummary,
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
            { withCredentials: true }
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
