import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

from haystack import Pipeline
from haystack.dataclasses import Document 
from haystack.document_stores.in_memory import InMemoryDocumentStore 
from haystack.components.preprocessors import DocumentSplitter 
from haystack.components.writers import DocumentWriter 
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder     
)
from haystack.utils import Secret
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever 
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator


try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY bulunamadı.")
        st.stop()
except Exception as e:
    st.error(f"Ortam değişkenleri yüklenirken bir hata oluştu: {e}")
    st.stop()

@st.cache_resource
def load_and_prepare_data():
    """
    Hulusiaa/tr_food_product_catalog_with_ingredients veri setini yükler,
    kolonları sabit eşler, Document'lere dönüştürür ve parçalara ayırır.
    """
    with st.spinner("Ürün kataloğu veri seti yükleniyor..."):
        try:
            ds = load_dataset("Hulusiaa/tr_food_product_catalog_with_ingredients", split="train")
            df = ds.to_pandas()

            if df.empty:
                st.error("Veri seti boş görünüyor.")
                return None

            # Sütun eşlemeleri
            COL_NAME       = "product_name"
            COL_BRAND      = "brand"
            COL_CATEGORY   = "category"
            COL_PRICE      = "price"
            COL_INGR_LIST  = "ingredientsList"
            COL_INGR_ALT   = "Ingredients"
            COL_ORIGIN     = "Origin"
            COL_WEIGHT     = "Net Weight (g/ml)"
            COL_USAGE      = "Usage Recommendations"
            COL_ALLERGEN   = "Allergen Warning"
            COL_STORAGE    = "Storage Conditions"
            COL_PRODUCER   = "Gıda İşletmecisi / Üretici / İthalatçı / Dağıtıcı"

            if COL_NAME not in df.columns:
                st.error(f"Gerekli sütun bulunamadı: {COL_NAME}")
                return None

            # Temizlik
            df_clean = df.dropna(subset=[COL_NAME]).copy()
            df_clean = df_clean[df_clean[COL_NAME].astype(str).str.strip() != ""]
            df_clean.reset_index(drop=True, inplace=True)

            documents = []
            for _, row in df_clean.iterrows():
                def val(col):
                    return str(row[col]).strip() if (col in df_clean.columns and pd.notna(row[col])) else ""

                name      = val(COL_NAME)
                brand     = val(COL_BRAND)
                category  = val(COL_CATEGORY)
                price     = val(COL_PRICE)
                ingr_list = val(COL_INGR_LIST)
                ingr_alt  = val(COL_INGR_ALT)
                origin    = val(COL_ORIGIN)
                weight    = val(COL_WEIGHT)
                usage     = val(COL_USAGE)
                allergen  = val(COL_ALLERGEN)
                storage   = val(COL_STORAGE)
                producer  = val(COL_PRODUCER)

                content_parts = [f"Ürün Adı: {name}"]
                if category:  content_parts.append(f"Kategori: {category}")
                if brand:     content_parts.append(f"Marka: {brand}")
                if price:     content_parts.append(f"Fiyat: {price}")
                if weight:    content_parts.append(f"Net Miktar: {weight}")
                if ingr_list: content_parts.append(f"İçindekiler (detay): {ingr_list}")
                elif ingr_alt: content_parts.append(f"İçindekiler: {ingr_alt}")
                if usage:     content_parts.append(f"Kullanım Önerileri: {usage}")
                if storage:   content_parts.append(f"Saklama Koşulları: {storage}")
                if allergen:  content_parts.append(f"Alerjen Uyarısı: {allergen}")
                if origin:    content_parts.append(f"Menşei: {origin}")
                if producer:  content_parts.append(f"İşletmeci/Üretici/İthalatçı/Dağıtıcı: {producer}")

                content = "\n\n".join(content_parts)

                meta = {
                    "name": name,
                    "brand": brand,
                    "category": category,
                    "price": price,
                    "weight": weight,
                    "origin": origin,
                }

                documents.append(Document(content=content, meta=meta))

            if not documents:
                st.error("Veri setinden belge üretilemedi.")
                return None

            splitter = DocumentSplitter(split_by="word", split_length=220, split_overlap=40)
            split_docs = splitter.run(documents)
            return split_docs["documents"]

        except Exception as e:
            st.error(f"Ürün veri seti hazırlanırken hata: {e}")
            return None


@st.cache_resource
def create_faiss_index(_split_docs):
    """
    Verilen belgeler için bir InMemory DocumentStore oluşturur ve doldurur.
    """
    if not _split_docs:
        return None
        
    with st.spinner("Vektör veritabanı oluşturuluyor ve belgeler işleniyor..."):
        try:
            document_store = InMemoryDocumentStore()
            
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="trmteb/turkish-embedding-model"
            )

            # Belgeleri ve gömme vektörlerini deposuna yazmak için bir boru hattı
            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("embedder", doc_embedder)
            indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
            indexing_pipeline.connect("embedder.documents", "writer.documents")

            # Boru hattını çalıştırarak indeksi oluştur
            indexing_pipeline.run({"embedder": {"documents": _split_docs}})
            
            return document_store
        except Exception as e:
            st.error(f"Vektör indeksi oluşturulurken hata oluştu: {e}")
            return None


@st.cache_resource
def build_rag_pipeline(_document_store):
    if not _document_store:
        return None

    try:
        if HF_TOKEN:
            text_embedder = SentenceTransformersTextEmbedder(
                model="trmteb/turkish-embedding-model",
                token=Secret.from_token(HF_TOKEN)
            )
        else:
            text_embedder = SentenceTransformersTextEmbedder(
                model="trmteb/turkish-embedding-model"
            )

        retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=6)

        # Chat için mesaj tabanlı şablon
        template = [
            ChatMessage.from_system(
                "Sen bir ürün katalog asistanısın. Yalnızca verilen belgelerdeki bilgilere dayanarak yanıt ver. "
                "Belgeler yetersizse 'Belgelerimde bu konu hakkında yeterli bilgi bulamadım.' de."
            ),
            ChatMessage.from_user(
                    """Ürünler:
                {% for doc in documents %}
                ---
                Belge (Ürün Parçası):
                {{ doc.content }}
                (Ürün: {{ doc.meta['name'] }} | Kategori: {{ doc.meta['category'] }} | Marka: {{ doc.meta['brand'] }} | Fiyat: {{ doc.meta['price'] }} | Net: {{ doc.meta['weight'] }} | Menşei: {{ doc.meta['origin'] }})
                ---
                {% endfor %}

                Soru: {{question}}"""
                            ),
                        ]


        prompt_builder = ChatPromptBuilder(
            template=template,
            required_variables=["question"]
        )

        generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_token(GOOGLE_API_KEY)
        )

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("generator", generator)

        # Bağlantılar
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "generator.messages") 
        

        return rag_pipeline

    except Exception as e:
        st.error(f"RAG boru hattı oluşturulurken hata oluştu: {e}")
        return None


def main():
    st.set_page_config(page_title="Ürün Arama Asistanı", page_icon="🛒")

    st.title("🛒 Türkiye Ürün Arama Asistanı")
    st.caption("Hugging Face: `Hulusiaa/tr_food_product_catalog_with_ingredients`")

    split_documents = load_and_prepare_data()
    if split_documents:
        document_store = create_faiss_index(split_documents)
        rag_pipeline = build_rag_pipeline(document_store) if document_store else None
    else:
        rag_pipeline = None

    if not rag_pipeline:
        st.warning("Uygulama başlatılamadı. Lütfen hata mesajlarını kontrol edin.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    placeholder = "Örn: Laktozsuz süt içeren atıştırmalık önerir misin?"
    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("İlgili ürünler taranıyor ve yanıt oluşturuluyor..."):
            try:
                result = rag_pipeline.run({
                    "text_embedder": {"text": prompt},
                    "prompt_builder": {"question": prompt}
                })

                response = "Bir hata oluştu veya yanıt alınamadı."
                if result and "generator" in result and result["generator"].get("replies"):
                    replies = result["generator"]["replies"]
                    if isinstance(replies, list) and len(replies) > 0:
                        chat_message = replies[0]
                        
                        # ChatMessage objesinden text çıkarma - daha güçlü yöntem
                        try:
                            if hasattr(chat_message, 'content') and isinstance(chat_message.content, list):
                                response = chat_message.content[0].text
                            elif hasattr(chat_message, 'content'):
                                response = str(chat_message.content)
                            else:
                                # String olarak çevir ve regex ile text'i çıkar
                                full_str = str(chat_message)
                                # text='...' kısmını bul
                                match = re.search(r"text='([^']*)'", full_str)
                                if match:
                                    response = match.group(1)
                                else:
                                    # text="..." formatını dene
                                    match = re.search(r'text="([^"]*)"', full_str)
                                    if match:
                                        response = match.group(1)
                                    else:
                                        response = full_str
                            
                            # Escape karakterleri düzelt
                            if isinstance(response, str):
                                response = response.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                                
                        except Exception as e:
                            # Son çare: manuel parsing
                            try:
                                full_str = str(chat_message)
                                match = re.search(r"text='([^']*)'", full_str)
                                if match:
                                    response = match.group(1).replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                                else:
                                    response = "Yanıt alınamadı."
                            except:
                                response = "Yanıt işlenirken hata oluştu."
                    else:
                        response = str(replies)

            except Exception as e:
                response = f"Sorgu işlenirken bir hata oluştu: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# Uygulamayı başlat
if __name__ == "__main__":
    main()