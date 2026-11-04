reelsProperties = {}

# ==================== A. 形式與互動架構 (Format & Delivery) ====================
# solo_facing: 主角全程直視鏡頭對話 (傳統單人口播)
# offscreen_dialogue: 藏鏡人/攝影師畫外發問，主角回應 (互動一問一答型)
# snippet_clip: Podcast 精華、講座切片、電視專訪節錄 (具第三方程式或現場感)
# dramatized_skit: 情境小劇場 (單人分飾兩角、或有對手戲演員)
# street_interview: 街訪路人或隨機挑戰
reelsProperties["presentation_style"] = "offscreen_dialogue"

# ==================== B. 內容切入載體 (Content Vehicle) ====================
# pure_knowledge: 純科普、生活經驗談、闢謠 (無特定實體載體)
# product_carrier: 藉由商業產品傳遞 (拆解某品牌保健品、零食成分表、外食聯名食品)
# demo_action: 焦點在食材實作/運動動作 (如減脂便當備餐示範、深蹲動作拆解)
# case_study: 個案分析 (拆解某個學員/名人的減肥案例與體檢報告)
reelsProperties["content_vehicle"] = "product_carrier"

# ==================== C. 核心研究焦點 (Focus Orientation) ====================
# mechanism_explainer: 生理/生化機制拆解 (如胰島素、皮質醇、腸道菌群)
# actionable_guide: 白話執行清單 (如超商外食指南、低卡替換原則)
# mindset_experience: 心態調適、減肥心路歷程
reelsProperties["content_focus"] = "mechanism_explainer"

# ==================== D. 製作與鏡頭語彙 (Production) ====================
reelsProperties["has_broll"] = True  # 是否穿插次要畫面 (食材、健身、外景)
reelsProperties["multi_angle_cut"] = 1  # 0: 單一固定機位 / 1: 多視角或頻繁跳剪縮放
reelsProperties["onscreen_text_heavy"] = (
    "keyword_highlight"  # full_subtitles / keyword_highlight
)
reelsProperties["visual_prop"] = (
    True  # 是否手持實物/道具 (產品外盒、模型、白板、平板)
)

# ==================== E. 敘事框架與說服技巧 (Narrative & Persuasion) ====================
reelsProperties["hook_type"] = (
    "myth_busting"  # pain_point / myth_busting / direct_action
)
reelsProperties["persona_tone"] = (
    "authoritative_strict"  # authoritative_strict / friendly_empathic / humorous_satirical
)
reelsProperties["professional_attire"] = False  # 是否穿著白袍/專業服裝
reelsProperties["background_setting"] = (
    "kitchen_gym"  # clinical_office / kitchen_gym / casual_indoor / studio
)

# ==================== F. 轉化與互動引導 (Conversion & CTA) ====================
reelsProperties["cta_type"] = (
    "comment_keyword"  # comment_keyword / discussion_prompt / none
)