import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Types based on API Documentation

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
    context_data: any;
    method: string;
}

// API Methods

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
};
