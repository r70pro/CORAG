# Medical Document Audits and Citations Rule

When generating timelines, audits, or summaries of medical documents (such as `souki_enclosures.pdf`):
1. **Never use raw system source tags** (like `[Source 26]` or `[Source 52]`) in final outputs.
2. **Always cite the exact page number range** of the original PDF document where the information is located.
3. **Include robust verification details** for each entry so that users can instantly verify the source when scrolling through the original file. This includes:
   - The exact document type and title (e.g., `Operation Record`, `Specialist Correspondence`).
   - The exact authoring physician or clinic (e.g., `Dr. Gavin Weekes`, `Capital Radiology`).
   - Identifying report details (e.g., `Ref No: 2024AL0008570-1`, `Accession Number: 77.50382801`).
