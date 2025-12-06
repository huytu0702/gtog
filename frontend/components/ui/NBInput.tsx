import React from 'react';
import { cn } from '@/lib/utils';

interface NBInputProps extends React.InputHTMLAttributes<HTMLInputElement> { }

export const NBInput = React.forwardRef<HTMLInputElement, NBInputProps>(
    ({ className, ...props }, ref) => {
        return (
            <input
                ref={ref}
                className={cn(
                    'w-full bg-white border-3 border-black p-3 font-medium placeholder:text-gray-500 focus:outline-none focus:shadow-hard transition-all',
                    className
                )}
                {...props}
            />
        );
    }
);

NBInput.displayName = 'NBInput';
