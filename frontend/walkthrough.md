# Walkthrough - Neo-Brutalism Frontend

I have successfully implemented the frontend for the GraphRAG system using the requested **Neo-Brutalism** design style.

## Features Implemented

### 1. Design System (Neo-Brutalism)
- **Theme**: High contrast, thick black borders (`3px`), hard shadows (`4px`), and vibrant colors (Lime Green, Pink).
- **Components**:
    - [NBButton](file:///f:/KL/gtog/frontend/components/ui/NBButton.tsx#9-13): Interactive buttons with hard shadow press effects.
    - [NBCard](file:///f:/KL/gtog/frontend/components/ui/NBCard.tsx#9-12): Container components with distinct borders and shadows.
    - [NBInput](file:///f:/KL/gtog/frontend/components/ui/NBInput.tsx#4-5): Bold input fields.
    - [NBLayout](file:///f:/KL/gtog/frontend/components/ui/NBLayout.tsx#9-46): Consistent layout with Navbar and Footer.

### 2. Collection Management (Dashboard)
- **List Collections**: View all knowledge bases in a grid layout.
- **Create Collection**: Simple form to create new collections.
- **Delete Collection**: Option to remove collections.

### 3. Document Management
- **Upload**: Interface to upload files ([.txt](file:///f:/KL/gtog/dictionary.txt), [.md](file:///f:/KL/gtog/README.md), etc.) to a collection.
- **List**: View uploaded documents with metadata.
- **Indexing**:
    - Real-time status tracking (Pending, Running, Completed, Failed).
    - Progress bar animation.
    - "Start Indexing" trigger.

### 4. Conversation Chat
- **Chat Interface**: A dedicated tab for interacting with the collection.
- **Search Methods**: Dropdown to select between Global, Local, ToG, and DRIFT search.
- **Conversational UI**: User (Right/Pink) and Bot (Left/Green) message bubbles.
- **State**: Handles loading states and errors gracefully.

## How to Run

1.  **Start the Backend** (if not already running):
    ```bash
    cd backend
    uvicorn app.main:app --reload
    ```

2.  **Start the Frontend**:
    ```bash
    cd frontend
    npm run dev
    ```

3.  **Access the App**:
    Open [http://localhost:3000](http://localhost:3000) in your browser.

## Verification Results

### Automated Build
- `npm run build` passed successfully.
- Type safety verified with TypeScript.
- Linting checks passed.

### Manual Verification Steps
1.  **Create a Collection**: Name it "Test Collection".
2.  **Upload a Document**: Upload a text file.
3.  **Index**: Click "Start Indexing" and watch the progress bar.
4.  **Chat**: Switch to the "Conversation Chat" tab and ask "What is this document about?".
