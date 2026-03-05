import React from 'react';
import Link from 'next/link';
import { EASY_AUTH_LOGIN_URL, EASY_AUTH_LOGOUT_URL } from '@/lib/api';

interface NBLayoutProps {
    children: React.ReactNode;
}

export function NBLayout({ children }: NBLayoutProps) {
    return (
        <div className="min-h-screen flex flex-col bg-bg text-text font-sans">
            {/* Navbar */}
            <header className="border-b-3 border-black bg-white sticky top-0 z-50">
                <div className="container mx-auto px-4 py-4 flex items-center justify-between">
                    <Link href="/" className="text-2xl font-black tracking-tighter hover:underline decoration-4 decoration-main">
                        GraphRAG<span className="text-main">.TOG</span>
                    </Link>
                    <nav className="flex items-center gap-4">
                        <Link href="/" className="font-bold hover:underline decoration-2">
                            Collections
                        </Link>
                        <a href="https://microsoft.github.io/graphrag/" target="_blank" rel="noopener noreferrer" className="font-bold hover:underline decoration-2">
                            Docs
                        </a>
                        <a
                            href={EASY_AUTH_LOGIN_URL}
                            className="px-3 py-2 border-3 border-black bg-main font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
                        >
                            Log In
                        </a>
                        <a
                            href={EASY_AUTH_LOGOUT_URL}
                            className="px-3 py-2 border-3 border-black bg-white font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
                        >
                            Log Out
                        </a>
                    </nav>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 container mx-auto px-4 py-8">
                {children}
            </main>

            {/* Footer */}
            <footer className="border-t-3 border-black bg-white py-8 mt-12">
                <div className="container mx-auto px-4 text-center font-bold">
                    <p>&copy; 2025 GraphRAG TOG.</p>
                </div>
            </footer>
        </div>
    );
}
