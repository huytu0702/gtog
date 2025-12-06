'use client';

import React from 'react';
import Link from 'next/link';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { collectionsApi } from '@/lib/api';
import { NBButton } from '@/components/ui/NBButton';
import { NBCard } from '@/components/ui/NBCard';
import { NBInput } from '@/components/ui/NBInput';
import { Plus, Trash2, FolderOpen, Loader2 } from 'lucide-react';

export default function Dashboard() {
  const queryClient = useQueryClient();
  const [newCollectionName, setNewCollectionName] = React.useState('');
  const [isCreating, setIsCreating] = React.useState(false);

  // Fetch Collections
  const { data, isLoading, error } = useQuery({
    queryKey: ['collections'],
    queryFn: collectionsApi.list,
  });

  // Create Collection Mutation
  const createMutation = useMutation({
    mutationFn: collectionsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] });
      setNewCollectionName('');
      setIsCreating(false);
    },
  });

  // Delete Collection Mutation
  const deleteMutation = useMutation({
    mutationFn: collectionsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] });
    },
  });

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCollectionName.trim()) return;
    createMutation.mutate({ name: newCollectionName });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-12 h-12 animate-spin text-main" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-destruct/10 border-3 border-destruct p-6 text-destruct font-bold">
        Error loading collections: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-5xl font-black mb-2">My Collections</h1>
          <p className="text-xl font-medium text-gray-600">
            Manage your knowledge bases and start chatting.
          </p>
        </div>
        <NBButton onClick={() => setIsCreating(!isCreating)} size="lg">
          <Plus className="w-6 h-6" />
          New Collection
        </NBButton>
      </div>

      {/* Create Form */}
      {isCreating && (
        <NBCard className="bg-white animate-in slide-in-from-top-4 fade-in duration-300">
          <h2 className="text-2xl font-bold mb-4">Create New Collection</h2>
          <form onSubmit={handleCreate} className="flex gap-4">
            <NBInput
              placeholder="Collection Name (e.g., product-docs)"
              value={newCollectionName}
              onChange={(e) => setNewCollectionName(e.target.value)}
              autoFocus
            />
            <NBButton type="submit" disabled={createMutation.isPending}>
              {createMutation.isPending ? 'Creating...' : 'Create'}
            </NBButton>
          </form>
        </NBCard>
      )}

      {/* Collections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {data?.collections.map((collection) => (
          <NBCard key={collection.id} className="group hover:bg-main/10 transition-colors relative">
            <div className="flex justify-between items-start mb-4">
              <div className="bg-main p-3 border-3 border-black shadow-hard-sm">
                <FolderOpen className="w-8 h-8" />
              </div>
              <button
                onClick={(e) => {
                  e.preventDefault();
                  if (confirm('Are you sure you want to delete this collection?')) {
                    deleteMutation.mutate(collection.id);
                  }
                }}
                className="p-2 hover:bg-destruct hover:text-white border-3 border-transparent hover:border-black transition-all rounded-none"
                title="Delete Collection"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>

            <Link href={`/collections/${collection.id}`} className="block">
              <h3 className="text-2xl font-bold mb-2 group-hover:underline decoration-2">
                {collection.name}
              </h3>
              <p className="text-gray-600 font-medium mb-4 line-clamp-2">
                {collection.description || 'No description provided.'}
              </p>

              <div className="flex items-center gap-4 text-sm font-bold">
                <span className="bg-white border-2 border-black px-2 py-1">
                  {collection.document_count} Docs
                </span>
                <span className={collection.indexed ? 'text-green-600' : 'text-gray-500'}>
                  {collection.indexed ? '● Indexed' : '○ Not Indexed'}
                </span>
              </div>
            </Link>
          </NBCard>
        ))}

        {data?.collections.length === 0 && !isCreating && (
          <div className="col-span-full text-center py-12 border-3 border-dashed border-gray-300">
            <p className="text-2xl font-bold text-gray-400 mb-4">No collections found.</p>
            <NBButton onClick={() => setIsCreating(true)} variant="outline">
              Create your first collection
            </NBButton>
          </div>
        )}
      </div>
    </div>
  );
}
