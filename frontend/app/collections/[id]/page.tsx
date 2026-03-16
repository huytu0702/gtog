'use client';

import React from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { collectionsApi, fetchEasyAuthUser } from '@/lib/api';
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

    const { data: user, isLoading: authLoading } = useQuery({
        queryKey: ['easy-auth-user'],
        queryFn: fetchEasyAuthUser,
        staleTime: 60_000,
        gcTime: 5 * 60_000,
        retry: false,
    });

    const { data: collection, isLoading, error } = useQuery({
        queryKey: ['collection', id],
        queryFn: () => collectionsApi.get(id),
        enabled: user?.isAuthenticated === true,
    });

    if (authLoading || (user?.isAuthenticated && isLoading)) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="w-12 h-12 animate-spin text-main" />
            </div>
        );
    }

    if (!user?.isAuthenticated) {
        return (
            <NBCard className="max-w-2xl bg-white">
                <h1 className="text-3xl font-black mb-3">Sign in to open collections</h1>
                <p className="text-lg font-medium text-gray-600 mb-4">
                    This screen only loads after Easy Auth confirms the current session.
                </p>
                <NBButton onClick={() => router.push('/')} variant="outline">
                    Go Back
                </NBButton>
            </NBCard>
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
