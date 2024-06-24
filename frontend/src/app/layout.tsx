import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "@/styles/global.css";
import React, { ReactNode } from "react";
import ErrorBoundary from './errorBoundary';  // Ensure correct import

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ArxFlix",
  description: "Create video from ArXiv papers",
};

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({
  children,
}: Readonly<RootLayoutProps>): JSX.Element {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
