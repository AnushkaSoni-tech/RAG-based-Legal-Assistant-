{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e0e2a020-27da-44dc-8f71-b6a7fde22dfc",
   "metadata": {},
   "source": [
    "## LOADING DOCUMENT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0ca81ade-641b-408a-b282-465be780ab56",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_community.document_loaders import PyPDFLoader\n",
    "loader = PyPDFLoader(\"consumer_act.pdf\")\n",
    "documents = loader.load()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d9a83abe-a36e-4ae3-a2ef-3af09a124a92",
   "metadata": {},
   "source": [
    "## CHUNKING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6d00cf15-5977-44f2-9596-af8d568c45b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#chunking\n",
    "from config import chunk_size , chunk_overlap\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter \n",
    "text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)\n",
    "chunks=text_splitter.split_documents(documents)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "57eb1de7-8255-4231-bbbf-059363328b38",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'producer': 'Microsoft® Word 2019', 'creator': 'Microsoft® Word 2019', 'creationdate': '2024-10-29T11:22:26+05:30', 'author': 'Lenovo', 'moddate': '2024-10-29T11:22:26+05:30', 'source': 'consumer_act.pdf', 'total_pages': 39, 'page': 0, 'page_label': '1'}\n",
      "{'producer': 'Microsoft® Word 2019', 'creator': 'Microsoft® Word 2019', 'creationdate': '2024-10-29T11:22:26+05:30', 'author': 'Lenovo', 'moddate': '2024-10-29T11:22:26+05:30', 'source': 'consumer_act.pdf', 'total_pages': 39, 'page': 0, 'page_label': '1'}\n"
     ]
    }
   ],
   "source": [
    "for chunk in chunks[:2]:\n",
    "    print(chunk.metadata)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6d065d7c-7f85-481a-a2bf-19f94ff5d962",
   "metadata": {},
   "source": [
    "# EMBEDDING"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55aa5568-ec93-4515-894e-26ad3230f697",
   "metadata": {},
   "source": [
    "## TFIDFVECTOR [SPARSE VECTPOR]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "62303f72-2561-4c17-8b65-b1264d0046da",
   "metadata": {},
   "outputs": [],
   "source": [
    "#dense - contexual meaning\n",
    "#tfidf - exact word\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "83fa5803-7e11-48cb-bb7a-1c1f53057d02",
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf_vec=TfidfVectorizer()\n",
    "tfidf_matrix=tfidf_vec.fit_transform([chunk.page_content for chunk in chunks])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ddb1ef1-932c-4d11-8702-cfdfd6feffb8",
   "metadata": {},
   "source": [
    "## DENSE VECTOR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d10bd457-23f8-4b85-afd0-08bc1c9a0741",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "from langchain_core.vectorstores import InMemoryVectorStore\n",
    "import google.generativeai as genai\n",
    "from config import embedding_model\n",
    "\n",
    "text=[doc.page_content for doc in chunks]\n",
    "\n",
    "from langchain_huggingface import HuggingFaceEmbeddings  \n",
    "\n",
    "#wrapped senttransformer inside huggingface bcoz vector store doesn't store numpy array and sentantranformer generate numpy array \n",
    "\n",
    "model =embedding_model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "909ed5e5-1973-475b-bc6e-a4cb0fd33a8a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['eca1ff4d-6116-4138-a0c2-827426d4afc2',\n",
       " '3072ef92-51c8-4bb1-9c28-8c1455a93503',\n",
       " '1beaa0da-c943-4cd5-88fe-1e7ffaadd5af',\n",
       " '22250882-f093-4db9-8916-2392df10ab0d',\n",
       " '07305e4f-61e5-4125-8c2c-4a780e676419',\n",
       " '4ee613f9-6945-40a8-b03d-710089663e99',\n",
       " '0caf494d-a346-4974-aafb-3afa0000b2dd',\n",
       " 'ff0a920c-1dd2-48f5-9f2f-6e45775f0e09',\n",
       " '196b8c33-bbec-405a-a668-b0150f0f6930',\n",
       " 'a586ab27-959e-495b-aeaf-56c303224e21',\n",
       " '682a7d28-17b7-47ef-bfdb-51f1fd2c53a6',\n",
       " 'f8213071-80de-4400-a92d-13296c232b48',\n",
       " 'b8529613-d6bc-43df-b304-10d163ea5ba4',\n",
       " 'd41197b5-d4af-4844-8fbb-1bbbfffbd4ce',\n",
       " 'ff42575a-f663-4d04-aa08-be067dc29329',\n",
       " '09e821be-fa32-424d-8171-4d39b9f947da',\n",
       " 'f6d832f6-fce1-4824-bcb6-e09cb649ec84',\n",
       " '064574b7-6b13-4040-ba7c-8c1311b3953a',\n",
       " 'd1bf172a-733f-4839-a226-714137fd2e3c',\n",
       " 'b414c30c-008f-4a15-a670-39f122806975',\n",
       " '8dda9968-3fee-4da0-a2b5-9873e23b7daf',\n",
       " '051f0890-4cc8-4a55-b0f3-76ec73add952',\n",
       " 'f0c13c09-c850-4512-88ed-6e2f97315c6a',\n",
       " '3c6bc8f3-02b2-4be5-83c2-b3f4239e7389',\n",
       " '388e1dc9-692c-4c94-b8b0-4a4ff6d4a13d',\n",
       " 'a0844719-7b40-4c7e-bfa6-bc336f12ba16',\n",
       " '824d2825-f771-4783-b779-bfcfe64e82f3',\n",
       " '9b06ab53-af89-42b4-9834-08610ecda37c',\n",
       " 'd9b4d790-14dc-4699-b8aa-3c0da2427155',\n",
       " '65f41cc4-c5bf-4cba-a0d5-3cd215829a81',\n",
       " '250f87bb-f641-4eb6-ab25-7d165def3326',\n",
       " '6bf90ded-f07e-4eed-b9ff-b4f210354719',\n",
       " 'd9d59871-9780-44e4-9a4b-f3c90652c024',\n",
       " '562f55a5-3527-4194-9621-a748b1be928d',\n",
       " '15eb935f-8512-4df0-a126-73cad16e1def',\n",
       " 'a49753ad-7e4b-4dc4-b289-1492d945522b',\n",
       " '8cc6974d-34df-424a-9282-ba85f6633c11',\n",
       " 'aa8e32c0-7584-474e-8496-ab83a851c826',\n",
       " '85b8821c-ec1f-48ef-8f8a-0ac2e939b107',\n",
       " 'b7f7c2ca-fcbe-41d0-8054-253de64b1ade',\n",
       " '8339691a-9025-4731-9fa7-fba5c707f4f7',\n",
       " '6d0aedf5-54f8-494e-ab34-a3e4d95b46ad',\n",
       " 'a349da48-a034-4bcf-9c18-204faf0e8f7c',\n",
       " 'f0e18878-e185-4c76-aa5c-e9bfd9cd9136',\n",
       " '0adcdce1-79f9-4b03-ba5b-176a2753c74f',\n",
       " 'c948d2c8-d47c-4aa8-8460-906f0d60feb2',\n",
       " 'ab92563d-9c51-4cb8-bc37-475f26ea04e2',\n",
       " 'a01c3217-790c-454f-ae53-88952a89e0ff',\n",
       " '08f7f42c-8893-4f55-abcf-0929c792e697',\n",
       " '006ad4e0-731d-4e31-9564-17b52b603f96',\n",
       " '97416249-17c9-454f-bd58-b53609617931',\n",
       " '1c4059b0-ff13-472e-a132-20cffc5c73b2',\n",
       " '801a8490-a238-41b5-8b9b-cdde52f66cc7',\n",
       " '8854b00b-69d7-45c4-b4e7-02665b154cb8',\n",
       " '92b39ae6-efdb-44ee-847b-07e07eb5de1f',\n",
       " '5b726dc1-9678-4766-a008-b6421a38258d',\n",
       " 'a9371a53-4a7a-4782-90c9-b4659d18807b',\n",
       " '623b0ea0-132a-4e2d-8ad3-f5f95899b434',\n",
       " '9eebce57-aeea-43da-892b-1395917bb112',\n",
       " '074fbbd0-2645-4fda-a7df-1bd2686c9995',\n",
       " '83ae0cdf-e796-4ea3-9400-cf0e965ef8e1',\n",
       " 'dfd84213-8888-4893-a5bc-121c7698aa16',\n",
       " 'e7bb0361-12ec-402e-b6be-32cbb09f1a0d',\n",
       " '7c512f29-83dc-4bf5-99ce-1e64bc64003d',\n",
       " 'c4d265ef-ef1a-4368-965e-8f9de8f37e84',\n",
       " '5f1ff5e4-18b9-4dc9-a88e-6e7b5cead049',\n",
       " '20c8c1c1-2f04-48e1-a30f-0deff0cb25ba',\n",
       " 'fddd1cbc-541c-4ccd-8ef1-3d31670449f9',\n",
       " 'a53b8d31-2ca2-449f-bfcc-20d206f3b143',\n",
       " '0b3a1a50-ba81-438c-98ae-8d0da99ae0a9',\n",
       " '91bda5d4-5398-4566-ab21-5951e3576544',\n",
       " '827725c8-0f60-4baa-bb68-1723c1d127fe',\n",
       " 'e9a8198c-b6f3-4181-8202-cd6455af8f01',\n",
       " '62d4b181-6326-4fd9-8210-26cbafa99c54',\n",
       " '0bc6a184-a6b9-4aa5-93d3-859cd0ca0aa4',\n",
       " '7e79b7bf-9195-4d75-8abc-314882ac3d3e',\n",
       " '42e9eb92-cb5b-4306-a5c5-52e1c559fa6d',\n",
       " '8ab24b40-6611-4740-adc0-4fe3cd6b1667',\n",
       " '34c8796e-8118-4a4f-a24b-af551b525081',\n",
       " 'a27ce47e-b439-4124-aa85-cba4662b8150',\n",
       " 'deb4316c-7c2b-40bc-8870-a27f27b435e5',\n",
       " 'b862a277-fc97-454d-bdd9-d3ab5a1cdd59',\n",
       " '5936e034-f9e8-4f53-841e-12a466364fd9',\n",
       " '73f3e779-14f8-41b3-9196-ad5d5722f7fc',\n",
       " '4fd08183-5a92-4e10-aea0-acd3f98c2d13',\n",
       " 'f89dce67-9122-4527-b91f-06eef26cf0e2',\n",
       " '2f1d0969-14d8-4c53-8413-07f3298fbb95',\n",
       " '93b94b6c-aa47-4277-bcfb-9cd79e6a3818',\n",
       " 'd7c07df1-87f6-4f77-a1b5-61273b8a0c7b',\n",
       " 'aa76a9ec-7305-4ca6-9ae5-547a245e858b',\n",
       " 'ef6204ca-3151-4156-babd-46a5117e9725',\n",
       " '36c6fe96-cd68-4925-b72b-d7fb7b20e0c0',\n",
       " '04dd174e-0aaa-416a-b059-14687af36c7e',\n",
       " '14b022c4-eb43-4f6b-82e5-58a27016e8c3',\n",
       " 'c0fb1997-05cf-4de5-939a-9e80f35838de',\n",
       " '3ab3e864-5825-4205-8e24-47499688e755',\n",
       " '30287d5f-2910-44d7-8f5e-88d082ceb650',\n",
       " '163a14a6-cf63-4ce5-8222-00bf76eb1897',\n",
       " '97f4c5f3-0e47-4d3e-86fb-3a8013de9d9b',\n",
       " '17d87ec6-70d5-450e-8766-9950afadf73f',\n",
       " '1de73ba6-02f4-4274-9748-a3ee09dbfbdb',\n",
       " '502fd7c5-96cf-4dcf-b7e5-d35e47d5550b',\n",
       " '7f3fdf58-97b1-4bbb-a222-34067a7a577d',\n",
       " 'ef08574d-27a4-4ceb-84e9-0c6397d2273f',\n",
       " 'bb817ac7-f73a-40dc-b59f-0b33cfcf1f28',\n",
       " 'e22d30df-2142-434e-9e5b-06c452ac3e77',\n",
       " 'a92b7d00-f4c5-44ca-9b49-bf6f53d36992',\n",
       " '44abf850-6441-4e01-8824-a62c6c37faa3',\n",
       " '1b7eb567-212a-49bf-9d82-8b2daf2cb6ba',\n",
       " '86bfff27-905f-466e-9c54-607702f7fc5e',\n",
       " 'c4c0d481-ec91-4636-892e-3bdbb281242b',\n",
       " 'fd4c32e8-eb46-4882-a744-2248869540f9',\n",
       " 'b38238f1-285e-4c8e-a41c-fc5ab953d985',\n",
       " '6dc5e428-5b42-4c6f-bb96-e9ced9a8201f',\n",
       " '0cd38e27-2c08-4055-8de3-8d53f6bbf213',\n",
       " '3cbfc6eb-dd7e-46b8-87ba-049ac0151b49',\n",
       " 'f8728546-7ec2-43bb-93eb-941fcc661415',\n",
       " 'd95cd2a5-14c4-4686-b230-5bbc72e707e5',\n",
       " 'b7837151-6545-48d8-b89d-46811efba57c',\n",
       " 'a3568503-0fbb-4325-ab77-b2465969eaae',\n",
       " '6fbe01cc-3b88-41a0-827c-fc53c6eeed73',\n",
       " 'a48b21d8-b7d7-4add-b4cd-d2ed92a5ff23',\n",
       " '97ef4a1e-40bd-4ff3-a68c-2d2032ba509b',\n",
       " '38501aff-1410-4f56-80fd-e2809c8c262b',\n",
       " 'a7394641-e23e-4c1f-b1fc-0b2e70dc6de5',\n",
       " '7fc7c369-9cc2-41a7-a52b-d4c5130953a5',\n",
       " '44856b50-97ed-4200-baae-62ee769adf3c',\n",
       " '1c7944c6-2500-45c0-8d40-03c8f61e4b43',\n",
       " '75e8877a-6b58-42eb-b372-5711d1d9604b',\n",
       " '65caf2bb-9b16-4d8b-af48-a4d4eeeba6c1',\n",
       " 'fb7bb041-b42a-4b69-bb05-c72ad967344e',\n",
       " '15987d9f-43f3-4fe6-b112-cbd4d5a18028',\n",
       " 'd7313e80-12fa-4b54-aa83-299d71394753',\n",
       " '6c41847c-881c-4263-854e-50c90c18a408',\n",
       " '313fd2c8-8e7a-4e28-bf3f-51ea37441fb1',\n",
       " '26290e6d-49de-4c1a-9da4-ca7ccf7ecf82',\n",
       " 'dc9f0117-0a60-43ae-b71c-79a1306406e8',\n",
       " '4b13f08a-3515-428a-a54c-a37bdc4c6952',\n",
       " '31df53bb-f9a8-450f-b2a3-9447b2a0c2dc',\n",
       " '1713ca9f-1a74-466b-8c69-83cd515ff5b1',\n",
       " 'e7febdd4-b7e8-4208-b49a-8d1057463ca1',\n",
       " '730c4e15-4b0f-4721-9ae8-81acc5c70c1d',\n",
       " 'c5f6f93d-ff40-43a6-a6e7-f382e180eb45',\n",
       " '428e7b0c-340f-45d5-800c-67b872dd4e9d',\n",
       " '6f3363ea-1466-4359-9c2d-a39707945b1e',\n",
       " 'd2b3b9ea-5669-4268-90ef-3385b76d3a61',\n",
       " 'be747490-a8f6-4301-bbfe-51c474696bea',\n",
       " '5ce1b99a-9054-4cf0-8491-3c0b5dc41808',\n",
       " 'fa107e62-8aed-4ab4-a8db-fa41ec6829c8',\n",
       " 'a21b3edc-dc98-4b1a-ab16-d7506f07c70a',\n",
       " '0f0a499e-d6a6-4b07-84ff-943fc1bd7ad8',\n",
       " '5c7b78b1-8820-4468-8fc9-5b3c3c3e9d00',\n",
       " '8b432efb-bc4b-4dde-9429-4e5a745cc86d',\n",
       " '8c053c01-d4d2-45ac-a519-d62049b06034',\n",
       " '5299a3fb-1591-4488-aa17-67c19179d38a',\n",
       " '5fe19131-b75d-4a96-8f83-eba05004b987',\n",
       " '6154b9a6-f896-48ea-b43c-bb28e681b783',\n",
       " '848c5441-53fe-4972-a72d-5a63b3d672c3',\n",
       " 'afe9dd2c-3f6a-4438-9536-529b080c5ee6',\n",
       " 'eb9bbbea-1f05-4a74-875f-922cc24c44b6',\n",
       " '3519d8cb-3f31-4e05-ba0a-ec86606ab990',\n",
       " '36723826-538d-4fc9-888a-5500ba004aa2',\n",
       " '1147ddf8-7db6-4956-8c4c-1bdf7220e634',\n",
       " '1475d5ca-d18f-4cc4-9b91-6d4f8f726bf6',\n",
       " '8bea28ef-d37d-4bc7-ac8a-85e2e6efc9f2',\n",
       " '8478b8a7-a75c-4fd1-92d9-9c7397471205',\n",
       " '7ac2deb0-2bb9-41e1-8f9b-45db407f168f',\n",
       " '8dc4ecaf-c696-4ddf-97fb-b138842fa73d',\n",
       " 'c5417f04-2127-4f42-8934-291c67faeb87',\n",
       " 'b0daa9d0-74fd-4a51-a6bb-985d9b3b28e3',\n",
       " '4c61d9b1-9847-44c8-acac-e3387149a3be',\n",
       " 'e7817977-600e-4c61-a6d7-97d46c374eb7',\n",
       " '12ad67df-17dc-42cb-b564-4d98d732371e',\n",
       " '03f97940-4aa6-4023-9ef6-c94ecaf0783e',\n",
       " '29917e21-5940-4e79-a66c-0f154f4bd00e',\n",
       " 'ee08356f-48c3-4905-9110-251c955e80f6',\n",
       " '30b3e51b-0e65-48a1-8f58-dfda0ba46935',\n",
       " 'b083c2e9-4f21-4aa3-876d-2bae541ba4c7',\n",
       " 'fb54f95a-5007-4dff-b2b5-2153dfd810a8',\n",
       " 'f912534c-7a67-46b2-8437-b6664e4b1548',\n",
       " 'aa6dfa05-7521-439f-8108-44d6c70d1a22',\n",
       " 'ef771d03-8483-47d3-aca7-58b1a901de7a',\n",
       " '9029a395-cf5c-400c-8fdf-bcb469801916',\n",
       " '4f0959aa-de69-424a-8046-bf5a4340c6cc',\n",
       " 'a68a0293-ee63-4213-b18e-b12f913c7075',\n",
       " '4d96ba2b-5414-4e78-bb16-9582b5a1528d',\n",
       " '43847df8-62a8-49cb-a06e-499bf8ed51e4',\n",
       " 'b728e006-d6b6-4428-94d1-a2d7de26e42e',\n",
       " 'f6bcf6b8-0e0e-4066-b563-71a9b93310bb',\n",
       " '85d8113a-ec5a-4c07-a8b9-28dc0fecef69',\n",
       " '1a99f602-7794-487d-9622-657d787744bc',\n",
       " '81e9d305-8997-413d-9d9c-f1a72bbb712c',\n",
       " '3a96a4af-0964-412e-9410-2ea010c39cbf']"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vectorstore=InMemoryVectorStore(embedding=model)\n",
    "vectorstore.add_documents(documents=chunks)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b136d07-9146-4e8e-bbcf-d34faa9ff9a4",
   "metadata": {},
   "source": [
    "## hybrid retrieval function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "32e80234-9482-4333-bec2-6d0cd870964e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def hybrid_retrival(query,k=4):\n",
    "    #dense\n",
    "    dense_doc=vectorstore.similarity_search(query,k=k)\n",
    "\n",
    "    #tfidf\n",
    "    query_=tfidf_vec.transform([query])\n",
    "    score=cosine_similarity(\n",
    "        query_,tfidf_matrix\n",
    "    )[0]\n",
    "\n",
    "    top_indices=score.argsort()[-k:][::1]  #indices of top k chunk\n",
    "    tfidf_doc=[chunks[i] for i in top_indices]\n",
    "\n",
    "    #combine\n",
    "    combine_doc=[]\n",
    "    for doc in dense_doc + tfidf_doc:\n",
    "        if doc not in combine_doc:\n",
    "            combine_doc.append(doc)\n",
    "    \n",
    "    return combine_doc[:k]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31a1cd3d-308e-4b2e-a1dc-e12ecf9b00ac",
   "metadata": {},
   "source": [
    "# creating prompt template"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78f43bad-4eff-4f55-9845-8828c43539e7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "5145b376-2481-4b46-ae3d-a7526190744c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      " recieved broken glass\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔎 Searching relevant legal provisions...\n",
      "\n",
      "AI: ### Issue\n",
      "You received a product described as \"broken glass,\" indicating potential damage to the product itself and possibly an \"injury.\"\n",
      "\n",
      "### Applicable Law\n",
      "The retrieved context defines \"injury\" as any harm illegally caused to any person, in body, mind or property [Source 1]. However, it explicitly states that \"harm caused to a product itself or any damage to the property on account of breach of warranty conditions or any commercial or economic loss, including any direct, incidental or consequential loss relating thereto\" is *not* included in the definition of \"harm\" for certain purposes [Source 1].\n",
      "\n",
      "A product liability action can be brought against a product seller under specific circumstances, such as if:\n",
      "*   The seller made an express warranty that the product failed to conform to, causing harm [Source 2].\n",
      "*   The product was sold, and the manufacturer's identity is unknown, or the manufacturer cannot be reached or enforced against [Source 2].\n",
      "*   The seller failed to exercise reasonable care in assembling, inspecting, or maintaining the product, or did not pass on manufacturer warnings, and this failure was the proximate cause of the harm [Source 2].\n",
      "\n",
      "There are exceptions where a product liability action cannot be brought against a product seller, for example, if the product was misused, altered, or modified at the time of harm [Source 3].\n",
      "\n",
      "A \"manufacturer\" is defined as a person who makes or assembles goods or parts, or puts their mark on goods made by others [Source 1].\n",
      "\n",
      "### Application to Your Situation\n",
      "If the \"broken glass\" refers to the product itself being damaged, the retrieved context indicates that \"harm caused to a product itself\" is generally excluded from the definition of \"harm\" for certain legal purposes [Source 1]. However, if the broken glass caused physical harm to you (injury to body or mind) or damage to other property you own, this could fall under the definition of \"injury\" [Source 1].\n",
      "\n",
      "To pursue a product liability claim against the product seller for the \"broken glass,\" you would generally need to show that one of the conditions outlined in the law is met. For instance, if the seller specifically warranted the product against breakage and it failed, or if the seller's lack of reasonable care in handling the product led to the damage [Source 2].\n",
      "\n",
      "However, if the damage occurred because you misused, altered, or modified the product, a product liability action against the seller might not be possible [Source 3].\n",
      "\n",
      "### Possible Remedy\n",
      "The retrieved legal provisions do not provide specific information about remedies like refunds, replacements, or compensation for a \"broken product\" in this general scenario. While the context discusses grounds for a \"product liability action,\" it does not detail the outcomes or specific forms of relief available in such an action.\n",
      "\n",
      "### Sources\n",
      "[Source 1][Source 2][Source 3]\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      " exit\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Hope it helped\n"
     ]
    }
   ],
   "source": [
    "import google.generativeai as genai\n",
    "from config import prompt\n",
    "genai.configure(api_key=\"AQ.Ab8RN6JawzVZKXz8XVvSXMmtFC6OHFa7mv5xpEgvzdLUAu7gQQ\")\n",
    "llm = genai.GenerativeModel(\"gemini-2.5-flash\")\n",
    "chat_history=[]\n",
    "while True:\n",
    "    question=input()\n",
    "    if question.lower() in[\"quit\",\"exit\",\"bye\"]:\n",
    "        break\n",
    "    retrieved_documents =hybrid_retrival(question,k=4)\n",
    "    # Create numbered context\n",
    "    retrieved_context = \"\\n\\n\".join(\n",
    "        [\n",
    "            f\"Source {i+1}:\\n{doc.page_content}\"\n",
    "            for i, doc in enumerate(retrieved_documents)\n",
    "        ]\n",
    "    )\n",
    "    #conversation history\n",
    "    history=\"\\n\".join(\n",
    "        [\n",
    "            f\"{message['role']}:{message['content']}\"\n",
    "            for message in chat_history\n",
    "        ]\n",
    "    )\n",
    "    #prompt\n",
    "    prompt=prompt.format(\n",
    "        history=history,\n",
    "        retrieved_context=retrieved_context,\n",
    "        question=question\n",
    "    )\n",
    "    print(\"🔎 Searching relevant legal provisions...\")\n",
    "    response=llm.generate_content(prompt)\n",
    "    answer=response.text\n",
    "    \n",
    "    print(\"\\nAI:\", answer)\n",
    "\n",
    "    #save conversation\n",
    "    chat_history.append(\n",
    "        {\n",
    "            \"role\": \"User\",\n",
    "            \"content\": question\n",
    "        }\n",
    "    )\n",
    "    chat_history.append(\n",
    "        {\n",
    "            \"role\": \"AI\",\n",
    "            \"content\": answer\n",
    "        }\n",
    "    )\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25f3c7fa-75cb-4f5d-8a2f-5f5d56f90c71",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
