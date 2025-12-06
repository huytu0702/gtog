'use client';

import React from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { collectionsApi } from '@/lib/api';
import { NBButton } from '@/components/ui/NBButton';
import { NBCard } from '@/components/ui/NBCard';
import { CollectionDocuments } from '@/components/collection-documents';
import { CollectionChat } from '@/components/collection-chat';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function CollectionDetailsPage() {
    const params = useParams();
    const router = useRouter();
    const id = params.id as string;
    const [activeTab, setActiveTab] = React.useState<'documents' | 'chat'>('documents');

    const { data: collection, isLoading, error } = useQuery({
        queryKey: ['collection', id],
        queryFn: () => collectionsApi.get(id),
    });

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="w-12 h-12 animate-spin text-main" />
            </div>
        );
    }

    if (error || !collection) {
        return (
            <div className="bg-destruct/10 border-3 border-destruct p-6 text-destruct font-bold">
                Error loading collection: {(error as Error)?.message || 'Collection not found'}
                <NBButton onClick={() => router.push('/')} variant="outline" className="mt-4 block">
                    Go Back
                </NBButton>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-4">
                <NBButton onClick={() => router.push('/')} variant="ghost" size="sm" className="px-2">
                    <ArrowLeft className="w-6 h-6" />
                </NBButton>
                <div>
                    <h1 className="text-4xl font-black">{collection.name}</h1>
                    {collection.description && (
                        <p className="text-gray-600 font-medium">{collection.description}</p>
                    )}
                </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b-3 border-black">
                <button
                    onClick={() => setActiveTab('documents')}
                    className={cn(
                        'px-8 py-3 font-bold text-lg border-t-3 border-x-3 border-black -mb-[3px] transition-all',
                        activeTab === 'documents'
                            ? 'bg-main text-black shadow-none'
                            : 'bg-gray-100 text-gray-500 hover:bg-gray-200 border-transparent hover:border-black'
                    )}
                >
                    Documents & Indexing
                </button>
                <button
                    onClick={() => setActiveTab('chat')}
                    className={cn(
                        'px-8 py-3 font-bold text-lg border-t-3 border-x-3 border-black -mb-[3px] transition-all ml-2',
                        activeTab === 'chat'
                            ? 'bg-secondary text-black shadow-none'
                            : 'bg-gray-100 text-gray-500 hover:bg-gray-200 border-transparent hover:border-black'
                    )}
                >
                    Conversation Chat
                </button>
            </div>

            {/* Content */}
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-300">
                {activeTab === 'documents' ? (
                    <CollectionDocuments collection={collection} />
                ) : (
                    <CollectionChat collection={collection} />
                )}
            </div>
        </div>
    );
}
