from annotated_types import doc


def build_page_index(doc, doc_type):

    pages = []

    for idx, page in enumerate(doc.pages):

        pages.append(
            {
                "page_number": idx + 1,
                "document_type": doc_type,
                "text": page.export_to_markdown()
            }
        )

    return pages