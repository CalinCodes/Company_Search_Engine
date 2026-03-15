# WRITEUP: Intent Qualification System (VibeHack 2026)

## 1. Approach (System Architecture)
Sistemul este conceput ca un **multistage funnel pipeline** (conductă de filtrare în etape), menit să proceseze volume mari de date brute și să livreze rezultate calificate printr-o analiză semantică profundă în etapa finală. 

### Componente și Interacțiune:
1.  **Etapa 1: Deconstrucția Intenției (The Parser)**: 
    * Utilizează modelul **DeepSeek-V3** pentru a transforma o interogare în limbaj natural într-un obiect JSON structurat.
    * Extrage filtre "dure" (țări, număr de angajați, venituri, coduri NAICS) și **cuvinte cheie semantice**.
    * Identifică **rolul de business** al țintei (ex: Furnizor, Competitor, Producător) pentru a rafina contextul căutării.
2.  **Etapa 1b: Filtrare Deterministă**:
    * Aplică filtrele extrase direct pe setul de date (`final_processed_data.json`). Această etapă reduce drastic spațiul de căutare folosind criterii de tip SQL, eliminând companiile care nu îndeplinesc constrângerile de bază (locație sau mărime).
3.  **Etapa 2: Recuperare Hibridă (The Ranker)**:
    * Clasifică candidații rămași folosind un scor combinat: **BM25** (pentru potriviri lexicale exacte) + **Similitudine Cosinus** pe embeddings (`all-MiniLM-L6-v2`).
    * Această metodă asigură capturarea termenilor tehnici specifici, cât și a conceptelor abstracte. 
4.  **Etapa 3: Filtru Final LLM (The Judge)**:
    * Cei mai buni `top_k` candidați sunt trimiși către **Qwen2.5-14B-Instruct**.
    * Modelul primește contextul complet al companiei (descriere, model de afaceri, piețe țintă) și decide dacă aceasta satisface *cu adevărat* intenția utilizatorului, oferind și o justificare.

---

## 2. Tradeoffs (Optimizări)
* **Acuratețe vs. Cost:** În loc să trimitem 500 de companii direct către un LLM (proces scump și lent), folosim Etapele 1 și 2 pentru a reduce lista la cei mai relevanți ~10-20 de candidați.
* **Viteză vs. Înțelegere:** Prin utilizarea modelelor locale de embedding și a bibliotecii `rank-bm25`, etapa de clasificare este extrem de rapidă. LLM-ul este rezervat doar pentru decizia finală "judgment-heavy".
* **Robustitate la Lipsa Datelor:** Pipeline-ul include o funcție de **Relaxare a Interogării**. Dacă Etapa 1 returnează 0 rezultate din cauza filtrelor prea stricte, sistemul elimină automat constrângerile (precum venitul) și reîncearcă căutarea bazându-se doar pe contextul semantic.

---

## 3. Error Analysis (Analiza Erorilor)
* **Dificultăți ale sistemului:**
    * **Ambiguitatea NAICS:** Unele companii au coduri NAICS generice. Dacă interogarea este foarte specifică, Etapa 1 s-ar putea să nu găsească un cod NAICS perfect, bazându-se excesiv pe Etapa 2 (semantică).
    * **Entități Multinaționale:** O companie poate fi înregistrată în Germania, dar să aibă activități de producție doar în Asia. Filtrul de țară (Etapa 1) ar putea să o păstreze, deși nu satisface o intenție de "producție locală".
* **Exemplu de Clasificare Greșită:** Un furnizor de software logistic ar putea trece de Etapa 2 pentru interogarea "Companii de logistică în România" deoarece descrierea conține cuvântul "logistică". Etapa 3 (Qwen) este menită să identifice această diferență de rol, dar dacă descrierea este vagă, LLM-ul poate produce un fals pozitiv.

---

## 4. Scaling (Scalabilitate)
Dacă sistemul ar trebui să gestioneze **100.000 de companii** per interogare:
1.  **Bază de Date Vectorială:** Înlocuirea filtrării în memorie cu o bază de date vectorială (ex: Qdrant, Pinecone sau Milvus) pentru Etapa 2.
2.  **Procesare Distribuită:** Procesarea în loturi (batching) a Etapei 3 (LLM) pe mai multe instanțe GPU în paralel.
3.  **Pre-indexare:** Calcularea embeddings-urilor offline (asincron) pentru întregul set de date, în loc de calculul la momentul interogării.

---

## 5. Failure Modes and Monitoring (Moduri de Eșec și Monitorizare)
* **Halucinații în Parsare:** Etapa 1 ar putea deduce greșit un cod NAICS sau o țară. 
    * *Soluție:* Monitorizarea `reasoning`-ului oferit de Parser și logarea cazurilor în care `total_output` în Etapa 1 este 0.
* **Latența API-ului:** Dependența de API-uri externe (Featherless) poate încetini experiența utilizatorului.
    * *Monitorizare:* Dashboard-uri pentru latența API-ului și alerte pentru limitele de rată (rate-limiting).

---

## 6. Critical Thinking (Gândire Critică)
* **Semnale Cheie:** Sistemul se bazează puternic pe calitatea câmpurilor `description` și `core_offerings`. Dacă acestea lipsesc, Etapele 2 și 3 devin mult mai puțin eficiente.
* **Suport Multilingv:** Implementarea include un strat de traducere (Google Translate API) care permite utilizatorilor să caute în limba maternă, în timp ce pipeline-ul rulează în engleză pentru consistență.