# EmotionAI: Real-Time Facial Emotion Recognition System

## 1. Project Overview

EmotionAI is a comprehensive, full-stack application designed to recognize and analyze human emotions from facial expressions in real-time. It leverages Deep Learning (CNN) for classification and a modern React/Vite frontend for a smooth user experience.

The system doesn't just provide a single label; it offers a detailed "Emotion Wheel" analysis, providing primary emotions, sub-emotions, and a ranked breakdown of all detected emotional states.

***

## 2. System Architecture

The project is divided into two main components:

- **Backend (Server):** A Flask-based Python server handling the machine learning inference.
- **Frontend (App):** A React/TypeScript application built with Vite, Tailwind CSS, and Shadcn UI.

### 2.1 Backend Implementation ([Server/app.py](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/Server/app.py))

- **Face Detection:** Uses OpenCV's Haar Cascade classifier ([haarcascade\_frontalface\_default.xml](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/Server/src/haarcascade_frontalface_default.xml)) to locate faces in a frame.
- **Emotion Classification:** A Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The model architecture consists of 4 convolutional layers, max-pooling, and dropout for regularization.
- **Emotion Wheel Logic:** The [detector.py](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/Server/src/detector.py) script maps raw FER-2013 labels (Happy, Sad, etc.) into a more complex hierarchy:
  - **Category:** Comfortable vs. Uncomfortable.
  - **Base Emotion:** 8 categories (Scared, Sad, Angry, Embarrassed, Happy, Loved, Confident, Neutral).
  - **Sub-Emotion:** Dynamic labels based on confidence thresholds (e.g., "Overwhelmed" vs. "Anxious").

### 2.2 Frontend Implementation ([App/src/App.tsx](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/App/src/App.tsx))

- **Real-Time Feed:** Captures webcam frames using a custom hook ([useCamera.ts](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/App/src/hooks/useCamera.ts)).
- **Emotion Display:** A sophisticated UI component ([EmotionDisplay.tsx](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/App/src/components/emotion/EmotionDisplay.tsx)) that features:
  - **Skeleton State:** A "Scanning" mode while the system is looking for a face.
  - **Stats Highlight:** Primary and sub-emotion details with confidence percentages.
  - **Breakdown Bars:** A ranked list of all detected emotions with visual progress bars.

***

## 3. Data Flow

1. **Input:** User's webcam captures a frame.
2. **Transmission:** The frame is sent as a Base64 string/Blob to the `/predict` endpoint.
3. **Processing:**
   - Server detects the face.
   - Crops, grayscales, and resizes the face to 48x48.
   - Passes the face through the CNN.
   - Generates the Wheel Base breakdown.
4. **Response:** Server returns a JSON object containing the top prediction and the full sorted list of all 8 emotions.
5. **Output:** React UI updates dynamically, rendering animations and stats.

***

## 4. Technology Stack & Resources

### 4.1 Backend (Inference Engine)

- **Python 3.8+**: Core programming language.
- **Flask**: Lightweight WSGI web application framework.
- **Flask-Cors**: Handling Cross-Origin Resource Sharing for frontend communication.
- **TensorFlow & Keras**: Deep learning frameworks used to load and run the pre-trained CNN model (`model.h5`).
- **OpenCV (opencv-python)**: Real-time computer vision library for face detection (Haar Cascades) and image preprocessing (Grayscale, Resize).
- **NumPy**: Numerical computing for array manipulation (image to tensor conversion).
- **FER-2013 Dataset**: The primary source of facial emotion data used for model training.

### 4.2 Frontend (User Interface)

- **React 18**: Component-based UI library.
- **TypeScript**: Adding static types to JavaScript for better development experience and safety.
- **Vite**: Modern frontend build tool for fast development and optimized builds.
- **Tailwind CSS**: Utility-first CSS framework for rapid UI styling.
- **Shadcn UI**: High-quality accessible components built with Radix UI and Tailwind CSS.
- **Radix UI**: Headless UI primitives used for complex components like Dialogs, Popovers, and Progress bars.
- **Lucide React**: Modern and consistent icon library.
- **TanStack Query (React Query)**: Powerful asynchronous state management for API interactions.
- **React Router Dom**: Client-side routing for the Detect, Home, and How It Works pages.
- **React Hook Form & Zod**: Form management and schema-based validation.
- **Recharts**: Charting library for visualizing the emotional breakdown data.
- **Sonner**: Opinionated toast component for notification handling.
- **Embla Carousel**: Lightweight carousel for showcasing features.

### 4.3 Development & DevOps Resources

- **ESLint**: Pluggable linting utility for JavaScript and TypeScript.
- **Vitest**: Vite-native unit test framework.
- **PostCSS & Autoprefixer**: Tools for transforming CSS with JavaScript and adding vendor prefixes.
- **Docker**: Containerization support via `Dockerfile` for standardized deployment.
- **Google Fonts (Inter & Space Grotesk)**: Modern typography for enhanced readability.

***

## 5. Results and Analysis

### 5.1 Backend Response (Structured Data)

The backend returns a comprehensive JSON object that provides a deep analysis of the detected emotion:

| Field                   | Type     | Description                                                                                           |
| :---------------------- | :------- | :---------------------------------------------------------------------------------------------------- |
| **emotion**             | `string` | The primary detected base emotion (e.g., "Scared", "Happy").                                          |
| **subEmotion**          | `string` | A granular emotion label based on confidence thresholds (e.g., "Overwhelmed").                        |
| **category**            | `string` | Categorization into "Comfortable" or "Uncomfortable".                                                 |
| **confidence**          | `number` | Percentage match for the top emotion.                                                                 |
| **fer13Label**          | `string` | The raw classification from the FER-2013 dataset (e.g., "Fear").                                      |
| **wheelBaseListSorted** | `Array`  | A ranked list of all 8 base emotions with their individual confidence levels and active sub-emotions. |

### 5.2 Frontend Result Visualization ([EmotionDisplay.tsx](file:///c%3A/Users/Asus/Desktop/project-hub-projects/Emotion-detection/App/src/components/emotion/EmotionDisplay.tsx))

The frontend translates this raw data into an interactive and highly visual dashboard:

1. **Skeleton Mode:** Before a face is detected, the component shows a "Scanning..." state with placeholder emotion names and 0% confidence, using a grayscale effect to indicate inactivity.
2. **Primary Insight:** Displays the top-ranked emotion in a large, bold format alongside its confidence percentage.
3. **Sub-Emotion Detail:** A secondary highlight box shows the specific `subEmotion` (e.g., "Overwhelmed") and the raw `fer13Label` for technical validation.
4. **Sub-Emotion Tags:** For the primary emotion, all possible sub-emotions are listed as tags, with the currently active one highlighted (e.g., in a red/destructive theme for uncomfortable states).
5. **Ranked Breakdown:** A full list of all 8 tracked emotions is displayed with dynamic progress bars. The colors of these bars switch between primary (blue) and destructive (red) based on whether the emotion is categorized as Comfortable or Uncomfortable.

### 5.3 Supported Emotions & Sub-Emotions

The system maps facial features to 8 core wheel-base emotions, each with its own confidence-based sub-emotions:

| Base Emotion    | Category      | Sub-Emotions (High → Low Confidence) |
| :-------------- | :------------ | :----------------------------------- |
| **Sad**         | Uncomfortable | Hurt, Disappointed, Lonely           |
| **Scared**      | Uncomfortable | Overwhelmed, Powerless, Anxious      |
| **Angry**       | Uncomfortable | Annoyed, Jealous, Bored              |
| **Embarrassed** | Uncomfortable | Ashamed, Excluded, Guilty            |
| **Happy**       | Comfortable   | Excited, Grateful, Caring            |
| **Neutral**     | Comfortable   | Creative, Calm, Relaxed              |
| **Loved**       | Comfortable   | Respected, Valued, Accepted          |
| **Confident**   | Comfortable   | Powerful, Brave, Hopeful             |

<br />

***

##

