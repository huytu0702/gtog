'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { buildEasyAuthLoginUrl, buildEasyAuthLogoutUrl, fetchEasyAuthUser, getEasyAuthUserLabel } from '@/lib/api';

export function AuthLinks() {
    const [redirectOrigin, setRedirectOrigin] = useState<string | undefined>(undefined);

    useEffect(() => {
        setRedirectOrigin(window.location.origin);
    }, []);


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
        <a
            href={buildEasyAuthLoginUrl(redirectOrigin)}
            className="px-3 py-2 border-3 border-black bg-main font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
        >
            Log In
        </a>
    );
}
