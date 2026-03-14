'use client';

import { useEffect, useState } from 'react';

import { buildEasyAuthLoginUrl, buildEasyAuthLogoutUrl } from '@/lib/api';

export function AuthLinks() {
    const [redirectOrigin, setRedirectOrigin] = useState<string | undefined>(undefined);

    useEffect(() => {
        setRedirectOrigin(window.location.origin);
    }, []);

    const loginUrl = buildEasyAuthLoginUrl(redirectOrigin);
    const logoutUrl = buildEasyAuthLogoutUrl(redirectOrigin);

    return (
        <>
            <a
                href={loginUrl}
                className="px-3 py-2 border-3 border-black bg-main font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
            >
                Log In
            </a>
            <a
                href={logoutUrl}
                className="px-3 py-2 border-3 border-black bg-white font-bold shadow-hard-sm hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-none transition-all"
            >
                Log Out
            </a>
        </>
    );
}
