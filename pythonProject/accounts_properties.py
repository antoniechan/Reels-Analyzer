accountProperties = {}

# 1. 組織實體形態 (Entity Type)
# personal: 個人自媒體 (單一固定面孔)
# team_brand: 診所、保健品品牌、健身房、團隊共用 (多固定成員輪流出鏡)
# media_curator: 網媒、自媒體街訪/採訪號 (無固定主角，專訪不同專家或路人)
accountProperties["entity_type"] = "personal"

# 2. 主理人專業背景構成 (Professional Background Matrix)
# 一個創作者可能身兼多職，建議採 Multi-label / List 形式記錄
accountProperties["credentials"] = [
    "professional",  # 營養師 / 醫師 / 藥師 (具法規門檻的醫事人員)
    "fitness_coach",  # 健身教練 / 體態管理師 / 運動選手
    "chef_foodie",  # 廚師 / 烘焙師 / 專業餐飲研發
    "psychologist",  # 心理諮商師 / 身心靈療癒
    "unrelated_career",  # 與健康無關的本職 (如工程師、全職媽媽、學生)
    "none_unclear",  # 背景不詳 / 無揭露
]

# 3. 帳號主題垂直度 (Niche Focus)
# pure_health: 100% 聚焦健康、營養、身材管理
# health_dominant: 70%+ 健康知識，偶穿插個人生活/Vlog
# lifestyle_hybrid: 兼具多領域 (如美妝+生活+偶爾講減脂)
# incidental: 帳號主軸非健康 (如美食探店、科技開箱)，僅偶爾出一支健康減肥短影音
accountProperties["niche_focus"] = "pure_health"
