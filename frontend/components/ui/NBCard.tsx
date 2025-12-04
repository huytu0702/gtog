import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface NBCardProps extends React.HTMLAttributes<HTMLDivElement> {
    noShadow?: boolean;
}

export const NBCard = React.forwardRef<HTMLDivElement, NBCardProps>(
    ({ className, noShadow = false, ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={cn(
                    'bg-white border-3 border-black p-6',
                    !noShadow && 'shadow-hard',
                    className
                )}
                {...props}
            />
        );
    }
);

NBCard.displayName = 'NBCard';
