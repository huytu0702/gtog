import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";
import { NBLayout } from "@/components/ui/NBLayout";

export const metadata: Metadata = {
  title: "GraphRAG UI",
  description: "Neo-Brutalism Interface for GraphRAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <NBLayout>
            {children}
          </NBLayout>
        </Providers>
      </body>
    </html>
  );
}
