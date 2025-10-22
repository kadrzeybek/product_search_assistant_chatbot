# TÜRK ÜRÜN ARAMA ASISTANI - RAG (Retrieval-Augmented Generation) UYGULAMASI

# Temel Python kütüphaneleri
import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

# Haystack AI framework - RAG pipeline için temel bileşenler
from haystack import Pipeline
from haystack.dataclasses import Document 
from haystack.document_stores.in_memory import InMemoryDocumentStore 
from haystack.components.preprocessors import DocumentSplitter 
from haystack.components.writers.document_writer import DocumentWriter 

# Embedding bileşenleri - metinleri vektöre çevirmek için
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder  # Dokümanları vektöre çevirir
from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder  # Kullanıcı sorgularını vektöre çevirir
from haystack.utils import Secret
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever 

# Chat ve AI generation bileşenleri
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Google Gemini API anahtarı ve Hugging Face token'ı yükleniyor
try:
    load_dotenv()  # .env dosyasından değişkenleri yükle
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Google Gemini için gerekli
    HF_TOKEN = os.getenv("HF_TOKEN")              # Hugging Face için opsiyonel
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY bulunamadı.")
        st.stop()
except Exception as e:
    st.error(f"Ortam değişkenleri yüklenirken bir hata oluştu: {e}")
    st.stop()

# VERİ YÜKLEME VE HAZIRLIK FONKSİYONU

@st.cache_resource  # Streamlit cache - bir kez yükleyip hafızada tut
def load_and_prepare_data():
    """
    Hugging Face'ten Türk gıda ürünleri veri setini yükler,
    kolonları sabit eşler, Document'lere dönüştürür ve parçalara ayırır.
    """
    with st.spinner("Ürün kataloğu veri seti yükleniyor..."):
        try:
            # Hugging Face'ten veri setini yükle
            ds = load_dataset("Hulusiaa/tr_food_product_catalog_with_ingredients", split="train")
            df = ds.to_pandas()

            if df.empty:
                st.error("Veri seti boş görünüyor.")
                return None

            # Sütun eşlemeleri - dataset kolonlarını tanımla
            COL_NAME       = "product_name"                           # Ürün adı
            COL_BRAND      = "brand"                                  # Marka
            COL_CATEGORY   = "category"                               # Kategori
            COL_PRICE      = "price"                                  # Fiyat
            COL_INGR_LIST  = "ingredientsList"                       # İçindekiler listesi
            COL_INGR_ALT   = "Ingredients"                           # Alternatif içindekiler
            COL_ORIGIN     = "Origin"                                # Menşei
            COL_WEIGHT     = "Net Weight (g/ml)"                     # Net ağırlık
            COL_USAGE      = "Usage Recommendations"                  # Kullanım önerileri
            COL_ALLERGEN   = "Allergen Warning"                      # Alerjen uyarısı
            COL_STORAGE    = "Storage Conditions"                    # Saklama koşulları
            COL_PRODUCER   = "Gıda İşletmecisi / Üretici / İthalatçı / Dağıtıcı"  # Üretici bilgisi

            if COL_NAME not in df.columns:
                st.error(f"Gerekli sütun bulunamadı: {COL_NAME}")
                return None

            # Veri temizliği - boş ve geçersiz kayıtları kaldır
            df_clean = df.dropna(subset=[COL_NAME]).copy()
            df_clean = df_clean[df_clean[COL_NAME].astype(str).str.strip() != ""]
            df_clean.reset_index(drop=True, inplace=True)

            # Haystack Document objelerine dönüştürme
            documents = []
            for _, row in df_clean.iterrows():
                # Güvenli değer alma fonksiyonu
                def val(col):
                    return str(row[col]).strip() if (col in df_clean.columns and pd.notna(row[col])) else ""

                # Tüm ürün bilgilerini çıkar
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

                # AI'ın okuyacağı text formatını oluştur
                content_parts = [f"Ürün Adı: {name}"]  # Her zaman ürün adı ile başla
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

                # Metadata - hızlı filtreleme için key bilgiler
                meta = {
                    "name": name,
                    "brand": brand,
                    "category": category,
                    "price": price,
                    "weight": weight,
                    "origin": origin,
                }

                # Haystack Document objesi oluştur
                documents.append(Document(content=content, meta=meta))

            if not documents:
                st.error("Veri setinden belge üretilemedi.")
                return None

            # Document splitting (chunking) - büyük dokümanları küçük parçalara böl
            # Bu, AI modellerin token limitlerini aşmamak ve daha iyi retrieval için önemli
            splitter = DocumentSplitter(split_by="word", split_length=220, split_overlap=40)
            split_docs = splitter.run(documents)
            return split_docs["documents"]

        except Exception as e:
            st.error(f"Ürün veri seti hazırlanırken hata: {e}")
            return None

# VEKTÖR VERİTABANI OLUŞTURMA FONKSİYONU

@st.cache_resource  # Bir kez oluştur, hafızada tut
def create_faiss_index(_split_docs):
    """
    Verilen belgeler için bir InMemory DocumentStore oluşturur ve doldurur.
    Bu fonksiyon dokümanları vektörlere çevirip aranabilir hale getirir.
    """
    if not _split_docs:
        return None
        
    with st.spinner("Vektör veritabanı oluşturuluyor ve belgeler işleniyor..."):
        try:
            # RAM'de çalışan hızlı document store
            document_store = InMemoryDocumentStore()
            
            # Türkçe metinler için optimize edilmiş embedding modeli
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="trmteb/turkish-embedding-model"  # Türkçe'ye özel model
            )

            # Belgeleri ve gömme vektörlerini deposuna yazmak için bir boru hattı
            # Pipeline: dokümanları al -> vektöre çevir -> veritabanına yaz
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

# RAG PIPELINE OLUŞTURMA FONKSİYONU

@st.cache_resource  # Bir kez oluştur, hafızada tut
def build_rag_pipeline(_document_store):
    """
    RAG (Retrieval-Augmented Generation) Pipeline oluşturur.
    Bu pipeline: kullanıcı sorusu -> benzer dokümanları bul -> AI ile yanıt üret
    """
    if not _document_store:
        return None

    try:
        # Kullanıcı sorgularını vektöre çeviren embedder
        # HF_TOKEN varsa authenticated erişim, yoksa public
        if HF_TOKEN:
            text_embedder = SentenceTransformersTextEmbedder(
                model="trmteb/turkish-embedding-model",
                token=Secret.from_token(HF_TOKEN)
            )
        else:
            text_embedder = SentenceTransformersTextEmbedder(
                model="trmteb/turkish-embedding-model"
            )

        # Benzer dokümanları bulan retriever (en benzer 6 doküman getir)
        retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=6)

        # Chat için mesaj tabanlı şablon - AI'ın nasıl davranacağını belirler
        template = [
            # Sistem mesajı: AI'ın rolü ve kuralları
            ChatMessage.from_system(
                "Sen bir ürün katalog asistanısın. Yalnızca verilen belgelerdeki bilgilere dayanarak yanıt ver. "
                "Belgeler yetersizse 'Belgelerimde bu konu hakkında yeterli bilgi bulamadım.' de."
            ),
            # Kullanıcı mesajı: dinamik olarak doldurulacak template
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

        # Template'i dinamik verilerle dolduran prompt builder
        prompt_builder = ChatPromptBuilder(
            template=template,
            required_variables=["question"]  # Zorunlu değişken: kullanıcı sorusu
        )

        # Google Gemini AI ile yanıt üreten generator
        generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",  # En son Gemini modeli
            api_key=Secret.from_token(GOOGLE_API_KEY)
        )

        # RAG Pipeline montajı - tüm bileşenleri birleştir
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)    # 1. Sorguyu vektöre çevir
        rag_pipeline.add_component("retriever", retriever)           # 2. Benzer dokümanları bul
        rag_pipeline.add_component("prompt_builder", prompt_builder) # 3. Prompt'u hazırla
        rag_pipeline.add_component("generator", generator)           # 4. AI yanıtı üret

        # Bağlantılar - veri akışını tanımla
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")  # Sorgu vektörü -> Retriever
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")       # Bulunan dokümanlar -> Prompt builder
        rag_pipeline.connect("prompt_builder.prompt", "generator.messages")           # Hazırlanan prompt -> AI generator
        
        return rag_pipeline

    except Exception as e:
        st.error(f"RAG boru hattı oluşturulurken hata oluştu: {e}")
        return None


# ANA UYGULAMA FONKSİYONU
def main():
    """
    Streamlit web uygulamasının ana fonksiyonu.
    Sayfa ayarları, veri yükleme ve chat arayüzünü yönetir.
    """
    # Streamlit sayfa konfigürasyonu
    st.set_page_config(page_title="Ürün Arama Asistanı", page_icon="🛒")

    # Ana başlık ve açıklama
    st.title("🛒 Türkiye Ürün Arama Asistanı")
    st.caption("🤖 **AI Destekli RAG Sistemi** | 📊 **Veri:** `Hulusiaa/tr_food_product_catalog_with_ingredients`")
    

    # Sistem başlatma süreci - veri yükleme ve pipeline oluşturma
    split_documents = load_and_prepare_data()           # 1. Veri setini yükle ve işle
    if split_documents:
        document_store = create_faiss_index(split_documents)  # 2. Vektör veritabanını oluştur
        rag_pipeline = build_rag_pipeline(document_store) if document_store else None  # 3. RAG pipeline'ı hazırla
    else:
        rag_pipeline = None

    # Sistem hazır değilse uygulamayı durdur
    if not rag_pipeline:
        st.warning("Uygulama başlatılamadı. Lütfen hata mesajlarını kontrol edin.")
        st.stop()

    # Chat geçmişi için session state yönetimi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Önceki chat mesajlarını göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Hızlı erişim butonları - örnek sorular için
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🥛 Bana Laktozsuz süt önerir misin?", use_container_width=True):
            st.session_state.sample_question = "Bana Laktozsuz süt önerir misin?"
    
    with col2:
        if st.button("� Gurmepack Pesto Fusilli 400 G fiyatı ne kadar?", use_container_width=True):
            st.session_state.sample_question = "Gurmepack Pesto Fusilli 400 G fiyatı ne kadar?"
    
    with col3:
        if st.button("🍫 Ülker Çikolatalı Gofretin içerğinde neler var listele", use_container_width=True):
            st.session_state.sample_question = "Ülker Çikolatalı Gofretin içerğinde neler var listele"
    
    # chat input alanı
    placeholder = "Sorunuzu buraya yazın... (Örn: Ülker markalı bisküviler)"
    
    # Örnek soru butonuna basıldıysa otomatik doldur
    default_value = st.session_state.get("sample_question", "")
    # Chat input ile soru alma
    if prompt := st.chat_input(placeholder):
        # Örnek soru state'ini temizle
        if "sample_question" in st.session_state:
            del st.session_state.sample_question
            
        # Kullanıcı mesajını kaydet ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Örnek soru butonuna basıldıysa otomatik işle
    elif "sample_question" in st.session_state and st.session_state.sample_question:
        prompt = st.session_state.sample_question
        del st.session_state.sample_question  # State'i temizle
        
        # Kullanıcı mesajını kaydet ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Eğer prompt varsa işleme devam et
    if 'prompt' in locals() and prompt:

        # RAG Pipeline ile yanıt üretme süreci
        with st.spinner("🔍 İlgili ürünler taranıyor ve AI yanıtı hazırlanıyor..."):
            try:
                # RAG Pipeline'ı çalıştır - en önemli adım!
                # Bu tek satır: sorguyu vektöre çevir -> benzer dokümanları bul -> AI yanıtı üret
                result = rag_pipeline.run({
                    "text_embedder": {"text": prompt},      # Kullanıcı sorgusunu vektöre çevir
                    "prompt_builder": {"question": prompt}  # Prompt template'ine soruyu ekle
                })

                # AI yanıtını çıkarma ve temizleme - güçlü parsing
                response = "Bir hata oluştu veya yanıt alınamadı."
                if result and "generator" in result and result["generator"].get("replies"):
                    replies = result["generator"]["replies"]
                    if isinstance(replies, list) and len(replies) > 0:
                        chat_message = replies[0]
                        
                        # ChatMessage objesinden text çıkarma
                        try:
                            # İlk olarak content attribute'unu kontrol et
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
                                    # text="..." formatı
                                    match = re.search(r'text="([^"]*)"', full_str)
                                    if match:
                                        response = match.group(1)
                                    else:
                                        response = full_str
                            
                            # Escape karakterleri düzeltir (\n, \t, \r)
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

        # AI yanıtını kaydet ve göster
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()  # Ana fonksiyonu çağır