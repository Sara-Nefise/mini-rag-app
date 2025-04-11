

from string import Template

#### RAG PROMPTLARI ####

#### Sistem ####

system_prompt = Template("\n".join([
    "Kullanıcıya yanıt oluşturmak için bir asistansınız.",
    "Kullanıcının sorgusuyla ilişkili bir dizi belge size sağlanacak.",
    "Sağlanan belgelere dayanarak bir yanıt oluşturmanız gerekiyor.",
    "Kullanıcının sorgusuyla ilgisi olmayan belgeleri dikkate almayın.",
    "Yanıt oluşturamıyorsanız kullanıcıdan özür dileyebilirsiniz.",
    "Yanıtı, kullanıcının sorgusunun dilinde oluşturmalısınız.",
    "Kullanıcıya karşı nazik ve saygılı olun.",
    "Yanıtınızda kesin ve öz olun. Gereksiz bilgilerden kaçının.",
]))

#### Belge ####
document_prompt = Template(
    "\n".join([
        "## Belge No: $doc_num",
        "### İçerik: $chunk_text",
    ])
)

#### Altbilgi ####
footer_prompt = Template("\n".join([
    "Yalnızca yukarıdaki belgelere dayanarak, lütfen kullanıcı için bir cevap oluşturun.",
    "## Soru:",
    "$query",
    "",
    "## Cevap:",
]))