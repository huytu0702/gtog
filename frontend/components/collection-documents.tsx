'use client';

import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi, indexingApi, Collection } from '@/lib/api';
import { NBButton } from '@/components/ui/NBButton';
import { NBCard } from '@/components/ui/NBCard';
import { NBInput } from '@/components/ui/NBInput';
import { Trash2, FileText, Upload, Play, RefreshCw, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CollectionDocumentsProps {
    collection: Collection;
}

export function CollectionDocuments({ collection }: CollectionDocumentsProps) {
    const queryClient = useQueryClient();
    const [file, setFile] = React.useState<File | null>(null);

    // Fetch Documents
    const { data: docsData, isLoading: docsLoading } = useQuery({
        queryKey: ['documents', collection.id],
        queryFn: () => documentsApi.list(collection.id),
    });

    // Fetch Indexing Status
    const { data: indexStatus, refetch: refetchIndex } = useQuery({
        queryKey: ['indexing', collection.id],
        queryFn: () => indexingApi.getStatus(collection.id),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === 'running' || status === 'pending' ? 2000 : false;
        },
    });

    // Upload Mutation
    const uploadMutation = useMutation({
        mutationFn: (file: File) => documentsApi.upload(collection.id, file),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents', collection.id] });
            setFile(null);
            // Reset file input
            const fileInput = document.getElementById('file-upload') as HTMLInputElement;
            if (fileInput) fileInput.value = '';
        },
    });

    // Delete Mutation
    const deleteMutation = useMutation({
        mutationFn: (name: string) => documentsApi.delete(collection.id, name),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents', collection.id] });
        },
    });

    // Indexing Mutation
    const indexMutation = useMutation({
        mutationFn: () => indexingApi.start(collection.id),
        onSuccess: () => {
            refetchIndex();
        },
    });

    const handleUpload = (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;
        uploadMutation.mutate(file);
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column: Documents List */}
            <div className="lg:col-span-2 space-y-6">
                <NBCard>
                    <div className="flex justify-between items-center mb-6">
                        <h2 className="text-2xl font-bold">Documents ({docsData?.total || 0})</h2>
                        <form onSubmit={handleUpload} className="flex gap-2">
                            <input
                                id="file-upload"
                                type="file"
                                className="hidden"
                                onChange={(e) => setFile(e.target.files?.[0] || null)}
                            />
                            <label
                                htmlFor="file-upload"
                                className="cursor-pointer bg-white border-3 border-black px-4 py-2 font-bold hover:bg-gray-100 flex items-center gap-2"
                            >
                                <Upload className="w-4 h-4" />
                                {file ? file.name : 'Select File'}
                            </label>
                            {file && (
                                <NBButton type="submit" size="sm" disabled={uploadMutation.isPending}>
                                    {uploadMutation.isPending ? 'Uploading...' : 'Upload'}
                                </NBButton>
                            )}
                        </form>
                    </div>

                    <div className="space-y-2">
                        {docsData?.documents.map((doc) => (
                            <div
                                key={doc.name}
                                className="flex items-center justify-between p-3 border-2 border-black bg-gray-50 hover:bg-white transition-colors"
                            >
                                <div className="flex items-center gap-3">
                                    <FileText className="w-6 h-6 text-gray-500" />
                                    <div>
                                        <p className="font-bold">{doc.name}</p>
                                        <p className="text-xs text-gray-500">
                                            {(doc.size / 1024).toFixed(2)} KB • {new Date(doc.uploaded_at).toLocaleDateString()}
                                        </p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => deleteMutation.mutate(doc.name)}
                                    className="text-destruct hover:bg-destruct/10 p-2 rounded"
                                    title="Delete"
                                >
                                    <Trash2 className="w-5 h-5" />
                                </button>
                            </div>
                        ))}
                        {docsData?.documents.length === 0 && (
                            <p className="text-center text-gray-500 py-8 italic">No documents uploaded yet.</p>
                        )}
                    </div>
                </NBCard>
            </div>

            {/* Right Column: Indexing Status */}
            <div className="space-y-6">
                <NBCard className="bg-main/10 border-main-accent">
                    <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <RefreshCw className="w-5 h-5" />
                        Indexing Status
                    </h2>

                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="font-bold">Status:</span>
                            <span
                                className={cn(
                                    'px-2 py-1 border-2 border-black font-bold text-sm uppercase',
                                    indexStatus?.status === 'completed'
                                        ? 'bg-green-400'
                                        : indexStatus?.status === 'running'
                                            ? 'bg-yellow-300 animate-pulse'
                                            : indexStatus?.status === 'failed'
                                                ? 'bg-red-400'
                                                : 'bg-gray-200'
                                )}
                            >
                                {indexStatus?.status || 'Pending'}
                            </span>
                        </div>

                        {indexStatus?.status === 'running' && (
                            <div className="w-full bg-white border-2 border-black h-4 rounded-full overflow-hidden">
                                <div
                                    className="bg-main h-full transition-all duration-500"
                                    style={{ width: `${indexStatus.progress}%` }}
                                />
                            </div>
                        )}

                        {indexStatus?.message && (
                            <p className="text-sm font-medium text-gray-700 bg-white/50 p-2 border-2 border-black/10">
                                {indexStatus.message}
                            </p>
                        )}

                        <NBButton
                            onClick={() => indexMutation.mutate()}
                            disabled={indexStatus?.status === 'running' || docsData?.total === 0}
                            className="w-full"
                            variant={indexStatus?.status === 'running' ? 'outline' : 'primary'}
                        >
                            {indexStatus?.status === 'running' ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" /> Indexing...
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4" /> Start Indexing
                                </>
                            )}
                        </NBButton>

                        {docsData?.total === 0 && (
                            <p className="text-xs text-destruct font-bold text-center">
                                Upload documents to start indexing.
                            </p>
                        )}
                    </div>
                </NBCard>
            </div>
        </div>
    );
}
