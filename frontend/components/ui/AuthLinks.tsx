'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import {
    buildEasyAuthLoginUrl,
    buildEasyAuthLogoutUrl,
    EASY_AUTH_LOGIN_PROVIDERS,
    fetchEasyAuthUser,
    getEasyAuthUserLabel,
} from '@/lib/api';

export function AuthLinks() {
    const [redirectOrigin, setRedirectOrigin] = useState<string | undefined>(undefined);
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const menuRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        setRedirectOrigin(window.location.origin);
    }, []);

    useEffect(() => {
        if (!isMenuOpen) {
            return;
        }

        const handlePointerDown = (event: MouseEvent) => {
            if (!menuRef.current?.contains(event.target as Node)) {
                setIsMenuOpen(false);
            }
        };

        const handleEscape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setIsMenuOpen(false);
            }
        };

        document.addEventListener('mousedown', handlePointerDown);
        document.addEventListener('keydown', handleEscape);

        return () => {
            document.removeEventListener('mousedown', handlePointerDown);
            document.removeEventListener('keydown', handleEscape);
        };
    }, [isMenuOpen]);

    const { data: user = null } = useQuery({
        queryKey: ['easy-auth-user'],
        queryFn: fetchEasyAuthUser,
        staleTime: 60_000,
        gcTime: 5 * 60_000,
        retry: false,
    });

    const logoutUrl = buildEasyAuthLogoutUrl(redirectOrigin);
    const userLabel = user ? getEasyAuthUserLabel(user) : null;

    if (user?.isAuthenticated) {
        return (
            <div className="flex items-center gap-3">
                {userLabel && (
                    <span className="max-w-52 truncate border-3 border-black bg-secondary px-3 py-2 font-bold shadow-hard-sm">
                        {userLabel}
                    </span>
                )}
                <a
                    href={logoutUrl}
                    className="px-3 py-2 border-3 border-black bg-white font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
                >
                    Log Out
                </a>
            </div>
        );
    }

    return (
        <div className="relative" ref={menuRef}>
            <button
                type="button"
                onClick={() => setIsMenuOpen((current) => !current)}
                aria-expanded={isMenuOpen}
                aria-haspopup="menu"
                className="px-3 py-2 border-3 border-black bg-main font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
            >
                Log In
            </button>
            {isMenuOpen && (
                <div className="absolute right-0 top-full z-50 mt-2 min-w-44 border-3 border-black bg-white p-2 shadow-hard-sm">
                    <div className="mb-2 px-2 text-xs font-bold uppercase tracking-wide text-gray-600">
                        Choose provider
                    </div>
                    <div className="flex flex-col gap-2">
                        {EASY_AUTH_LOGIN_PROVIDERS.map((provider) => (
                            <a
                                key={provider.id}
                                href={buildEasyAuthLoginUrl(provider.id, redirectOrigin)}
                                className="border-3 border-black bg-white px-3 py-2 font-bold hover:bg-main transition-colors"
                            >
                                {provider.label}
                            </a>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
