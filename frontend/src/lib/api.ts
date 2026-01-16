export interface QueryResponse {
    answers: Array<{
        question: string;
        result: {
            answer: string;
            extracted_facts: Array<{
                fact: string;
                evidence_ids: string[];
            }>;
            insufficiency_note?: string;
            confidence: string;
            evidence_texts?: string[];
            used_evidence?: string[];
        };
    }>;
    counts: {
        nodes: number;
        edges: number;
    };
    timing_s: {
        total: number;
        graph_build: number;
        index_build: number;
        qa_total: number;
    };
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export async function queryFromFile(
    file: File,
    query: string,
    options: any = {}
): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append("pdf", file);
    formData.append("questions", JSON.stringify([query]));
    formData.append("options", JSON.stringify(options));

    const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API error: ${response.statusText}`);
    }

    return response.json();
}

export async function queryFromUrl(
    url: string,
    query: string,
    options: any = {}
): Promise<QueryResponse> {
    const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            url: url,
            questions: [query],
            options: options,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API error: ${response.statusText}`);
    }

    return response.json();
}

export async function getHealth() {
    const response = await fetch(`${API_URL}/health`);
    return response.json();
}
