import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface NBButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'destructive' | 'outline' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
}

export const NBButton = React.forwardRef<HTMLButtonElement, NBButtonProps>(
    ({ className, variant = 'primary', size = 'md', ...props }, ref) => {
        const variants = {
            primary: 'bg-main hover:bg-main-accent text-black',
            secondary: 'bg-secondary hover:bg-secondary-accent text-black',
            destructive: 'bg-destruct text-white hover:opacity-90',
            outline: 'bg-white hover:bg-gray-50 text-black',
            ghost: 'bg-transparent border-transparent shadow-none hover:bg-gray-100',
        };

        const sizes = {
            sm: 'px-3 py-1.5 text-sm',
            md: 'px-6 py-3 text-base',
            lg: 'px-8 py-4 text-lg',
        };

        const baseStyles = 'font-bold border-3 border-black shadow-hard transition-all active:translate-x-[2px] active:translate-y-[2px] active:shadow-hard-sm disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-2';

        return (
            <button
                ref={ref}
                className={cn(baseStyles, variants[variant], sizes[size], className)}
                {...props}
            />
        );
    }
);

NBButton.displayName = 'NBButton';
