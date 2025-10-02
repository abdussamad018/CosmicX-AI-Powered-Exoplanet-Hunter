
import { StrictMode } from 'react'
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from 'react-query'
import { ReactQueryDevtools } from 'react-query/devtools'
import { Toaster } from 'sonner'
import App from "./App.tsx";
import "./index.css";
import { queryClient } from './lib/react-query'

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <Toaster position="top-right" />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </StrictMode>
);
  