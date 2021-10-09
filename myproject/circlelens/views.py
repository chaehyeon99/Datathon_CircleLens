from django.shortcuts import render
import random
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

def contents_based(cn):
  contents = pd.read_csv('circlelens/contents.csv',encoding='cp949')
  contents_rm_name = contents.drop(['동아리'], axis = 1)
  c_sim = cosine_similarity(contents_rm_name).argsort()[:,::-1]

  def get_recommend_club_list(df, club_name, top=30):
    # 해당 동아리에 대한 index 정보를 뽑아 냄.
    target_club_index = contents[contents['동아리'] == club_name].index.values

    # 비슷한 코사인 유사도를 가진 정보를 뽑아 냄.
    sim_index = c_sim[target_club_index, :top].reshape(-1)

    # 본인 제외
    sim_index = sim_index[sim_index != target_club_index]

    # dataframe으로 만든 뒤 return
    result = df.iloc[sim_index][:3]['동아리']
    return result

  final = get_recommend_club_list(contents, cn).tolist()
  return final


def item_based(cn):
    ratings = pd.read_csv('circlelens/ratings.csv', encoding='cp949')
    contents = pd.read_csv('circlelens/contents.csv', encoding='cp949')

    item_user_ratings = ratings.pivot_table('ratings', index='club', columns='userid')  # 데이터 형태 변경
    item_user_ratings.fillna(0, inplace=True)

    item_based_collabor = cosine_similarity(item_user_ratings)  # 유사도 측정 척도로 cosine similarity를 사용
    clubid = item_user_ratings.index

    item_based_collabor = pd.DataFrame(data=item_based_collabor, index=contents['동아리'], columns=contents['동아리'])

    def get_item_based_collabor(title):
        rec = item_based_collabor[title].sort_values(ascending=False)[1:4]
        top_item_titles = rec.index.values.tolist()
        return top_item_titles

    final = get_item_based_collabor(cn)
    return final



def classification(receive):
  data = pd.read_csv("circlelens/data.csv", encoding = 'cp949')

  X = data.drop(['동아리명'], axis = 1)
  y = data['동아리명']


  # 데이터 컬럼 단위 정규화 하기
  # 각 변수들의 max,min 값이 다르기 때문에 값이 큰 변수들일수록 모델에 영향을 크게 미치기 때문에 모든 변수의 범위를 일정하게 조절

  CatBoost = CatBoostClassifier(random_seed = 1234,
                                max_depth = 9,
                                n_estimators = 150,
                                learning_rate = 0.6)

  CatBoost.fit(X, y, verbose = False)

  columns = ['학업에의 영향', '동아리활동 소요시간', '친밀도', '동아리방 시설', '술문화',
        '기대했던 활동 유무', '신입부원과 친해지는 정도', '동아리 평균 활동비', '친목행사', '자유로운 동아리방 사용',
        '수평적인 분위기', '소속감', '코로나19 대비 방안마련', '활동적인 부원들', '학교외 활동', '초심자를 위한 활동',
        '동아리 장비가 갖춰진 정도', '소모임, 스터디', '원하는 활동 진행', '외부 지원', '학과구성', '행사규모',
        '활동기간', '회비', '추천하는 학년_1', '추천하는 학년_2', '추천하는 학년_3', '추천하는 학년_4',
        '활동 강제 정도_대부분 필참', '활동 강제 정도_반드시 한 번 이상 필참', '활동 강제 정도_자유',
        '동아리원 실력_골고루 섞여있음', '동아리원 실력_모두', '동아리원 실력_아마추어', '동아리원 실력_입문자',
        '동아리원 실력_프로 수준', '활동 빈도_격주에 한 번', '활동 빈도_매주 한 번 이상', '활동 빈도_정해져 있지 않음',
        '활동 빈도_한 학기에 한 번', '활동 빈도_한달에 한 번', '활동 장소_자유로움', '활동 장소_캠퍼스 내부',
        '활동 장소_캠퍼스 외부', '공통 주제_선택 활동 분야/주제가 있음', '공통 주제_없음',
        '공통 주제_필수 활동 분야/주제가 있음', '성비_남자 비율이 높음', '성비_비슷함', '성비_여자 비율이 높음',
        '선발절차_1차', '선발절차_2차', '선발절차_없음', '공연', '봉사', '자기계발', '취미', '인문과학', '종교',
        '체육']

  practice = []
  for i in range(60):
      practice.append(0)

  practice[0:24] = receive[0:24]

  practice_data = pd.DataFrame(practice, index = columns)
  practice_data = practice_data.transpose()


  if receive[24] == "1학년":
    practice_data['추천하는 학년_1'][0] = 1
  if receive[24] == "2학년":
    practice_data['추천하는 학년_2'][0] = 1
  if receive[24] == "3학년":
    practice_data['추천하는 학년_3'][0] = 1
  if receive[24] == "4학년":
    practice_data['추천하는 학년_4'][0] = 1

  if receive[25] == "대부분 필참":
    practice_data['활동 강제 정도_대부분 필참'][0] = 1
  if receive[25] == "반드시 한 번 이상 필참":
    practice_data['활동 강제 정도_반드시 한 번 이상 필참'][0] = 1
  if receive[25] == "자유":
    practice_data['활동 강제 정도_자유'][0] = 1

  if receive[26] == "골고루 섞여 있음":
    practice_data['동아리원 실력_골고루 섞여있음'][0] = 1
  if receive[26] == "입문자":
    practice_data['동아리원 실력_입문자'][0] = 1
  if receive[26] == "아마추어":
    practice_data['동아리원 실력_아마추어'][0] = 1
  if receive[26] == "프로 수준":
    practice_data['동아리원 실력_프로 수준'][0] = 1

  if receive[27] == "매주 한 번 이상":
    practice_data['활동 빈도_매주 한 번 이상'][0] = 1
  if receive[27] == "격주에 한 번":
    practice_data['활동 빈도_격주에 한 번'][0] = 1
  if receive[27] == "한달에 한 번":
    practice_data['활동 빈도_한달에 한 번'][0] = 1
  if receive[27] == "한 학기에 한 번":
    practice_data['활동 빈도_한 학기에 한 번'][0] = 1
  if receive[27] == "정해져 있지 않음":
    practice_data['활동 빈도_정해져 있지 않음'][0] = 1

  if receive[28] == "자유로움":
    practice_data['활동 장소_자유로움'][0] = 1
  if receive[28] == "캠퍼스 내부":
    practice_data['활동 장소_캠퍼스 내부'][0] = 1
  if receive[28] == "캠퍼스 외부":
    practice_data['활동 장소_캠퍼스 외부'][0] = 1

  if receive[29] == "필수 활동 분야/주제가 있음":
    practice_data['공통 주제_필수 활동 분야/주제가 있음'][0] = 1
  if receive[29] == "선택 활동 분야/주제가 있음":
    practice_data['공통 주제_선택 활동 분야/주제가 있음'][0] = 1
  if receive[29] == "없음":
    practice_data['공통 주제_없음'][0] = 1

  if receive[30] == "남자 비율이 높음":
    practice_data['성비_남자 비율이 높음'][0] = 1
  if receive[30] == "비슷함":
    practice_data['성비_비슷함'][0] = 1
  if receive[30] == "여자 비율이 높음":
    practice_data['성비_여자 비율이 높음'][0] = 1

  if receive[31] == "1차":
    practice_data['선발절차_1차'][0] = 1
  if receive[31] == "2차":
    practice_data['선발절차_2차'][0] = 1
  if receive[31] == "없음":
    practice_data['선발절차_없음'][0] = 1

  if receive[32] == "공연":
    practice_data['공연'][0] = 1
  if receive[32] == "봉사":
    practice_data['봉사'][0] = 1
  if receive[32] == "자기계발":
    practice_data['자기계발'][0] = 1
  if receive[32] == "취미":
    practice_data['취미'][0] = 1
  if receive[32] == "인문과학":
    practice_data['인문과학'][0] = 1
  if receive[32] == "종교":
    practice_data['종교'][0] = 1
  if receive[32] == "체육":
    practice_data['체육'][0] = 1

  classification_result = CatBoost.predict(practice_data)[0][0]
  club_name = ['KURP', '고란도란', '불교학생회', 'KULAX', '아마추어축구부', '고대바둑사랑', 'KUBT', 'KUCC', '유스호스텔', '고고쉼', '원불교학생회', 'SFC',
               'CCC', '뇌의주름', 'ENM', '돌빛', '고집', '수호회', '고전기타부', 'KUSEP', 'KUBC', 'FC엘리제', '고려대학교 관현악단', '노래얼', '백구회',
               '농구연합회', '한일문화연구회', '그루터기', '운화회', '한량회', '택견한울', '한국화회', 'TTP', '소믈리에', '호영회', '크림슨', '고려대학교 관악부',
               'ATP', '고대농악대', 'JASS', '서화회', '궁도회', '캘리쿠', '팝콘', '거의격월간몰라도되는데', '그림마당', '호우회', 'LoGS', '열두루달', 'ALC',
               '고려대학교 합창단', '한국사회연구회', 'ECS', '중국연구회', '예술비평연구회', '뉴런', '불아스', '한국근현대사연구회', 'TERRA', '국악연구회', '미스디렉션',
               'UNSA', 'KUDT', '극예술연구회', '수레바퀴', 'IVF', '로타랙트', '젊은예수', '평화나비', '예수전도단', 'KURC', '호진회', 'KUSA',
               '소울메이트', 'JOY', 'ENTHES', '탁구사랑회', '고풋', 'LECA', '사람과 사람', '철학마을', '고대문학회']
  club_f = ['#세미나 #강연회 #토론', '#토론 #논쟁 #화합', '#법회 #불교행사 #친목', '#운동 #팀스포츠 #리그', '#유일무이축구중앙동아리, #축구',
            '#바둑 #대회 #교류전 #리그전', '#자덕 #자율 #힐링 ', '#컴퓨터,#코딩,#개발', '#여행 #추억 #MT', '#고양이, #동물권, #봉사',
            '#마음공부, #명상, #mindfulness', '#학생신앙운동, #신앙고백공동체, #나눔', '#CCC #기독교동아리 #서울북CCC #사랑 #크리스천', '#보드게임 #취미 ',
            '#기독교, #제자훈련, #세계선교', '#영화#드라마#예술', '#집고치기, #건축봉사, #해비타트', '#Sports, #체육분과, #수영하는 호랑이, ',
            '#클래식기타 #고기부잡으로줌으로갈까요(랜선회식) #정기연주회', '#환경보호 #환경캠페인 #환경동아리', '#배드민턴, #운동, #사랑', '#아마추어 #여자축구부 #fc엘리제',
            '#아마추어오케스트라, #고대음대, #연주회', '#밴드 #공연 #친목', '#야구, #근본, #OneTeamOneSpirit',
            '#basketball #team sports #뒷풀이 냠냠', '#서브컬쳐 #JLPT #소모임', '#통기타 #어쿠스틱 밴드 #오디션 없음', '#반디공부방 #교육봉사 #52년 전통',
            '#국궁 #활 #전통무예', '#고려대택견 #호신술 #운동', '#붓과 먹 #도란도란 #힐링', '#피아노 #연주회 #친목활동', '#와인 #페어링 #테이스팅', '#카메라 #사진 #출사',
            '#중앙락밴드, #열정적인, #화목한', '#금관악기 #목관악기 #타악기 #앙상블', '#음악 #버스킹 #가족같은', '#이런거 #살면서언제 #해보겠어',
            '#감상모임 #합주 #째애즈에_칵테일_한_잔', '#서예 #서양화 #예술혼', '#양궁 #고대양궁 #궁도회 #체육분과 #고려대학교 #고려대학교양궁동아리', '#손글씨 #예쁜글귀 #엽서',
            '#광고, #아이디어, #마케팅', '#기획 및 제작(창작) #판매', '#만화 #그림 #초심자도 괜찮아요', '# 화목한분위기 #현충원봉사 #멘토링 #실버복지',
            '#아카펠라 #Acappella #음악', '#생태탐사 #자연 #생물', '#영어회화, #연대연합', '#합창 #화음 #친목', '#세미나 #기행 #쉼표', '#영어회화#친목#연극',
            '#중국어 #한자 #동아시아', '#예술 #친목 #가입제한없음', '#코로나?_어림도없지#물음표살인마도_환영하는#닫을문도없는_열린뇌과학학회!', '#댄스스포츠 #누구나출수있다 #최고급동방',
            '#사회, #독서, #대화', '#힙합, #R&B, #공연', '#국악, #공연', '#마술, #공연, #친목', '#국제 이슈, #토론, #유엔', '#춤#스트릿댄스#공연',
            '#연극 #공연 #이무대의주인공은당신', '#사회문제 #사회과학학회 #연대활동 ', '#기독교동아리, #복음주의, #누구나함께', '# 알찬봉사 # 다양한 프로그램',
            '#유대감, #신앙, #여유', "#일본군'위안부' #평화 #인권", '#사랑이넘치는 #가족 #흘러가는사랑 #열방을품다', '#봉사, #친목, #자율참여', '# 영화 # 이야기',
            '#사회분과#봉사#KUSA', '#순수_창작_뮤지컬, #대학의_로망, #좋아서_하는_개고생', '#기독교, #말씀짝, #성경스케치', '#친목 #운동 #활기참',
            '#탁구 #sports #상시모집', '#고풋, #풋살, #futsal', '#친목 #경험 #문화', '#퀴어, #성소수자, #인권', '#학문 #철학 #사색',
            '#문학 #예술 #S급동방']

  club_tag = club_f[club_name.index(classification_result)]
  classification_result = [classification_result, club_tag]
  return classification_result


def new_recom(dataframe):
    dataframe = pd.DataFrame(dataframe)
    u = dataframe['userid'][0]
    c = dataframe['club']
    r = dataframe['ratings']

    # 데이터 불러오기
    contents = pd.read_csv('circlelens/contents.csv', encoding='cp949')
    ratings = pd.read_csv('circlelens/ratings.csv', encoding='cp949')
    matching = pd.read_csv('circlelens/matching.csv', encoding='cp949')

    cid = []
    for i in c:
        name = int(matching[matching['동아리'] == i]['동아리번호'])
        cid.append(name)

    dataframe['club'] = cid
    ratings.index = range(ratings.shape[0])
    dataframe.index = range(ratings.shape[0], (ratings.shape[0] + 10))
    ratings = pd.concat([ratings, dataframe])

    reader = Reader(rating_scale=(1, 5))  # 선호도는 1점~5점
    data = Dataset.load_from_df(ratings[['userid', 'club', 'ratings']], reader=reader)
    full_train = data.build_full_trainset()  # 전체 추천 결과에 전체 데이터셋 사용

    item_user_ratings = ratings.pivot_table('ratings', index='club', columns='userid')  # 데이터 형태 변경
    item_user_ratings.fillna(0, inplace=True)
    item_based_collabor = cosine_similarity(item_user_ratings)
    clubid = item_user_ratings.index
    matching = pd.concat([pd.DataFrame(contents['동아리'], index=range(82)), pd.DataFrame(clubid, index=range(82))],
                         axis=1)  # 동아리 번호와 동아리 이름 매칭을 위한 DataFrame

    ########################################
    # 추천하는 함수
    def recomm_items(algo, userid, top_n=False, unseen=True, sorting=True):

        # 평점 여부에 따라 list 생성
        ## 전체 item id
        total_items = matching['club'].tolist()
        ## 평점을 내린 item id
        seen_items = ratings[ratings['userid'] == userid]['club'].tolist()
        ## 평점을 내리지 않은 item id
        unseen_items = [items for items in total_items if items not in seen_items]

        # 평점을 내리지 않은 item에 대해 평점 예측하는 경우
        if unseen:
            predictions = [algo.predict(userid, itemId) for itemId in unseen_items]

        # 평점을 이미 내린 item에 대해 평점 예측하는 경우
        else:
            predictions = [algo.predict(userid, itemId) for itemId in seen_items]

        # 예측평점(est) 기준으로 정렬하는 함수
        def sortkey_est(pred):
            return pred.est

        # 정렬하는 경우
        if sorting:
            predictions.sort(key=sortkey_est, reverse=True)

        # 상위 n개 결과만을 도출하는 경우
        if top_n:
            predictions = predictions[:top_n]

        top_item_ids = [pred.iid for pred in predictions]
        top_item_ratings = [pred.est for pred in predictions]

        cnt = 0
        for i in top_item_ids:
            name = matching[matching['club'] == i]['동아리']
            if cnt == 0:
                top_item_titles = name
                cnt += 1
            else:
                top_item_titles = pd.concat([top_item_titles, name])

        top_item_preds = [(ids, rating, title) for ids, rating, title in
                          zip(top_item_ids, top_item_ratings, top_item_titles)]

        return top_item_preds

    ##########################################

    # 학습
    algo1 = SVD(n_factors=10, n_epochs=5, random_state=42)
    algo1.fit(full_train)

    algo2 = SVDpp(n_factors=30, n_epochs=5, random_state=42)
    algo2.fit(full_train)

    algo3 = SVDpp(n_factors=50, n_epochs=50, random_state=42)
    algo3.fit(full_train)

    ################################################
    # 이미 평점을 매긴 데이터가 train data로 사용되므로, 해당 데이터셋 build하는 함수 생성
    def build_train(userid):
        seen_items_preds_SVD = recomm_items(algo1, userid, top_n=False, unseen=False, sorting=False)
        seen_items_preds_SVDpp = recomm_items(algo2, userid, top_n=False, unseen=False, sorting=False)
        seen_items_preds_NMF = recomm_items(algo3, userid, top_n=False, unseen=False, sorting=False)

        SVD_ratings = []
        for i in range(len(seen_items_preds_SVD)):
            SVD_ratings.append(seen_items_preds_SVD[i][1])

        SVDpp_ratings = []
        for i in range(len(seen_items_preds_SVDpp)):
            SVDpp_ratings.append(seen_items_preds_SVDpp[i][1])

        NMF_ratings = []
        for i in range(len(seen_items_preds_NMF)):
            NMF_ratings.append(seen_items_preds_NMF[i][1])

        seen_ratings = ratings[ratings['userid'] == userid]['ratings'].tolist()
        train = pd.DataFrame({"SVD": SVD_ratings, "SVDpp": SVDpp_ratings, "NMF": NMF_ratings, "true": seen_ratings})

        return train

    # 전체 user에 대한 train data 구축
    cnt = 0
    for user in ratings['userid'].unique():
        new = build_train(user)
        if cnt == 0:
            train = new
            cnt += 1
        else:
            train = pd.concat([train, new])

    #############################################
    # 평점을 매기지 않은 동아리에 대한 예측 평점 데이터를 test data로 사용
    def build_test(userid):

        ## 전체 item id
        total_items = matching['club'].tolist()
        ## 평점을 내린 item id
        seen_items = ratings[ratings['userid'] == userid]['club'].tolist()
        ## 평점을 내리지 않은 item id
        unseen_items = [items for items in total_items if items not in seen_items]

        all_items_preds_SVD = recomm_items(algo1, userid, top_n=False, unseen=True, sorting=False)
        all_items_preds_SVDpp = recomm_items(algo2, userid, top_n=False, unseen=True, sorting=False)
        all_items_preds_NMF = recomm_items(algo3, userid, top_n=False, unseen=True, sorting=False)

        clubid = []
        SVD_ratings = []
        SVDpp_ratings = []
        NMF_ratings = []

        for i in range(len(all_items_preds_SVD)):
            clubid.append(all_items_preds_SVD[i][2])
            SVD_ratings.append(all_items_preds_SVD[i][1])
            SVDpp_ratings.append(all_items_preds_SVDpp[i][1])
            NMF_ratings.append(all_items_preds_NMF[i][1])

        test = pd.DataFrame(
            {"userid": [userid] * len(clubid), "clubid": clubid, "SVD": SVD_ratings, "SVDpp": SVDpp_ratings,
             "NMF": NMF_ratings})

        return test

    ################################################
    test = build_test(u)

    y = train['true']
    X = train.drop(['true'], axis=1)

    # X,y 전체(train set 전체)로 학습
    X_train = X
    y_train = y

    gbm = ensemble.GradientBoostingRegressor()
    gbm.fit(X_train, y_train)

    # 평점이 없는 동아리 예측
    x_test = test[['SVD', 'SVDpp', 'NMF']]
    test_pred = gbm.predict(x_test)

    df1 = pd.DataFrame(test[['userid', 'clubid']])
    df1.index = list(range(test.shape[0]))
    df2 = pd.DataFrame(test_pred, columns=['pred'])

    gbm_pred = pd.concat([df1, df2], ignore_index=True, axis=1)
    gbm_pred.columns = ['userid', 'clubid', 'pred']
    gbm_pred['pred'] = round(gbm_pred['pred'], 2)
    gbm_pred  # 평점을 매기지 않은 모든 동아리에 대한 예측 평점

    gbm_pred = gbm_pred.sort_values(by='pred', axis=0, ascending=False)[:5]  # 상위 5개

    club_list = gbm_pred['clubid'].tolist()
    ratings_list = gbm_pred['pred'].tolist()
    club_name = ['KURP', '고란도란', '불교학생회', 'KULAX', '아마추어축구부', '고대바둑사랑', 'KUBT', 'KUCC', '유스호스텔', '고고쉼', '원불교학생회', 'SFC',
                 'CCC', '뇌의주름', 'ENM', '돌빛', '고집', '수호회', '고전기타부', 'KUSEP', 'KUBC', 'FC엘리제', '고려대학교 관현악단', '노래얼', '백구회',
                 '농구연합회', '한일문화연구회', '그루터기', '운화회', '한량회', '택견한울', '한국화회', 'TTP', '소믈리에', '호영회', '크림슨', '고려대학교 관악부',
                 'ATP', '고대농악대', 'JASS', '서화회', '궁도회', '캘리쿠', '팝콘', '거의격월간몰라도되는데', '그림마당', '호우회', 'LoGS', '열두루달', 'ALC',
                 '고려대학교 합창단', '한국사회연구회', 'ECS', '중국연구회', '예술비평연구회', '뉴런', '불아스', '한국근현대사연구회', 'TERRA', '국악연구회', '미스디렉션',
                 'UNSA', 'KUDT', '극예술연구회', '수레바퀴', 'IVF', '로타랙트', '젊은예수', '평화나비', '예수전도단', 'KURC', '호진회', 'KUSA',
                 '소울메이트', 'JOY', 'ENTHES', '탁구사랑회', '고풋', 'LECA', '사람과 사람', '철학마을', '고대문학회']
    club_f = ['#세미나 #강연회 #토론', '#토론 #논쟁 #화합', '#법회 #불교행사 #친목', '#운동 #팀스포츠 #리그', '#유일무이축구중앙동아리, #축구',
              '#바둑 #대회 #교류전 #리그전', '#자덕 #자율 #힐링 ', '#컴퓨터,#코딩,#개발', '#여행 #추억 #MT', '#고양이, #동물권, #봉사',
              '#마음공부, #명상, #mindfulness', '#학생신앙운동, #신앙고백공동체, #나눔', '#CCC #기독교동아리 #서울북CCC #사랑 #크리스천', '#보드게임 #취미 ',
              '#기독교, #제자훈련, #세계선교', '#영화#드라마#예술', '#집고치기, #건축봉사, #해비타트', '#Sports, #체육분과, #수영하는 호랑이, ',
              '#클래식기타 #고기부잡으로줌으로갈까요(랜선회식) #정기연주회', '#환경보호 #환경캠페인 #환경동아리', '#배드민턴, #운동, #사랑', '#아마추어 #여자축구부 #fc엘리제',
              '#아마추어오케스트라, #고대음대, #연주회', '#밴드 #공연 #친목', '#야구, #근본, #OneTeamOneSpirit',
              '#basketball #team sports #뒷풀이 냠냠', '#서브컬쳐 #JLPT #소모임', '#통기타 #어쿠스틱 밴드 #오디션 없음', '#반디공부방 #교육봉사 #52년 전통',
              '#국궁 #활 #전통무예', '#고려대택견 #호신술 #운동', '#붓과 먹 #도란도란 #힐링', '#피아노 #연주회 #친목활동', '#와인 #페어링 #테이스팅', '#카메라 #사진 #출사',
              '#중앙락밴드, #열정적인, #화목한', '#금관악기 #목관악기 #타악기 #앙상블', '#음악 #버스킹 #가족같은', '#이런거 #살면서언제 #해보겠어',
              '#감상모임 #합주 #째애즈에_칵테일_한_잔', '#서예 #서양화 #예술혼', '#양궁 #고대양궁 #궁도회 #체육분과 #고려대학교 #고려대학교양궁동아리', '#손글씨 #예쁜글귀 #엽서',
              '#광고, #아이디어, #마케팅', '#기획 및 제작(창작) #판매', '#만화 #그림 #초심자도 괜찮아요', '# 화목한분위기 #현충원봉사 #멘토링 #실버복지',
              '#아카펠라 #Acappella #음악', '#생태탐사 #자연 #생물', '#영어회화, #연대연합', '#합창 #화음 #친목', '#세미나 #기행 #쉼표', '#영어회화#친목#연극',
              '#중국어 #한자 #동아시아', '#예술 #친목 #가입제한없음', '#코로나?_어림도없지#물음표살인마도_환영하는#닫을문도없는_열린뇌과학학회!', '#댄스스포츠 #누구나출수있다 #최고급동방',
              '#사회, #독서, #대화', '#힙합, #R&B, #공연', '#국악, #공연', '#마술, #공연, #친목', '#국제 이슈, #토론, #유엔', '#춤#스트릿댄스#공연',
              '#연극 #공연 #이무대의주인공은당신', '#사회문제 #사회과학학회 #연대활동 ', '#기독교동아리, #복음주의, #누구나함께', '# 알찬봉사 # 다양한 프로그램',
              '#유대감, #신앙, #여유', "#일본군'위안부' #평화 #인권", '#사랑이넘치는 #가족 #흘러가는사랑 #열방을품다', '#봉사, #친목, #자율참여', '# 영화 # 이야기',
              '#사회분과#봉사#KUSA', '#순수_창작_뮤지컬, #대학의_로망, #좋아서_하는_개고생', '#기독교, #말씀짝, #성경스케치', '#친목 #운동 #활기참',
              '#탁구 #sports #상시모집', '#고풋, #풋살, #futsal', '#친목 #경험 #문화', '#퀴어, #성소수자, #인권', '#학문 #철학 #사색',
              '#문학 #예술 #S급동방']

    club_tag =[]
    for i in club_list :
        club_tag.append(club_f[club_name.index(i)])

    result = []
    for (i, j,k ) in zip(club_list, ratings_list, club_tag):
        result.append([i, j, k])

    return result

def index(request) :
    return render(request, "circlelens\index.html")

def reward(request) :
    club_list = pd.read_csv('circlelens/club_tag.csv', encoding='cp949')
    club_list.columns = ["동아리명", "동아리번호", "동아리 특색 해시태그"]
    club_name = ['KURP', '고란도란', '불교학생회', 'KULAX', '아마추어축구부', '고대바둑사랑', 'KUBT', 'KUCC', '유스호스텔', '고고쉼', '원불교학생회', 'SFC', 'CCC', '뇌의주름', 'ENM', '돌빛', '고집', '수호회', '고전기타부', 'KUSEP', 'KUBC', 'FC엘리제', '고려대학교 관현악단', '노래얼', '백구회', '농구연합회', '한일문화연구회', '그루터기', '운화회', '한량회', '택견한울', '한국화회', 'TTP', '소믈리에', '호영회', '크림슨', '고려대학교 관악부', 'ATP', '고대농악대', 'JASS', '서화회', '궁도회', '캘리쿠', '팝콘', '거의격월간몰라도되는데', '그림마당', '호우회', 'LoGS', '열두루달', 'ALC', '고려대학교 합창단', '한국사회연구회', 'ECS', '중국연구회', '예술비평연구회', '뉴런', '불아스', '한국근현대사연구회', 'TERRA', '국악연구회', '미스디렉션', 'UNSA', 'KUDT', '극예술연구회', '수레바퀴', 'IVF', '로타랙트', '젊은예수', '평화나비', '예수전도단', 'KURC', '호진회', 'KUSA', '소울메이트', 'JOY', 'ENTHES', '탁구사랑회', '고풋', 'LECA', '사람과 사람', '철학마을', '고대문학회']
    club_f = ['#세미나 #강연회 #토론', '#토론 #논쟁 #화합', '#법회 #불교행사 #친목', '#운동 #팀스포츠 #리그', '#유일무이축구중앙동아리, #축구', '#바둑 #대회 #교류전 #리그전', '#자덕 #자율 #힐링 ', '#컴퓨터,#코딩,#개발', '#여행 #추억 #MT', '#고양이, #동물권, #봉사', '#마음공부, #명상, #mindfulness', '#학생신앙운동, #신앙고백공동체, #나눔', '#CCC #기독교동아리 #서울북CCC #사랑 #크리스천', '#보드게임 #취미 ', '#기독교, #제자훈련, #세계선교', '#영화#드라마#예술', '#집고치기, #건축봉사, #해비타트', '#Sports, #체육분과, #수영하는 호랑이, ', '#클래식기타 #고기부잡으로줌으로갈까요(랜선회식) #정기연주회', '#환경보호 #환경캠페인 #환경동아리', '#배드민턴, #운동, #사랑', '#아마추어 #여자축구부 #fc엘리제', '#아마추어오케스트라, #고대음대, #연주회', '#밴드 #공연 #친목', '#야구, #근본, #OneTeamOneSpirit', '#basketball #team sports #뒷풀이 냠냠', '#서브컬쳐 #JLPT #소모임', '#통기타 #어쿠스틱 밴드 #오디션 없음', '#반디공부방 #교육봉사 #52년 전통', '#국궁 #활 #전통무예', '#고려대택견 #호신술 #운동', '#붓과 먹 #도란도란 #힐링', '#피아노 #연주회 #친목활동', '#와인 #페어링 #테이스팅', '#카메라 #사진 #출사', '#중앙락밴드, #열정적인, #화목한', '#금관악기 #목관악기 #타악기 #앙상블', '#음악 #버스킹 #가족같은', '#이런거 #살면서언제 #해보겠어', '#감상모임 #합주 #째애즈에_칵테일_한_잔', '#서예 #서양화 #예술혼', '#양궁 #고대양궁 #궁도회 #체육분과 #고려대학교 #고려대학교양궁동아리', '#손글씨 #예쁜글귀 #엽서', '#광고, #아이디어, #마케팅', '#기획 및 제작(창작) #판매', '#만화 #그림 #초심자도 괜찮아요', '# 화목한분위기 #현충원봉사 #멘토링 #실버복지', '#아카펠라 #Acappella #음악', '#생태탐사 #자연 #생물', '#영어회화, #연대연합', '#합창 #화음 #친목', '#세미나 #기행 #쉼표', '#영어회화#친목#연극', '#중국어 #한자 #동아시아', '#예술 #친목 #가입제한없음', '#코로나?_어림도없지#물음표살인마도_환영하는#닫을문도없는_열린뇌과학학회!', '#댄스스포츠 #누구나출수있다 #최고급동방', '#사회, #독서, #대화', '#힙합, #R&B, #공연', '#국악, #공연', '#마술, #공연, #친목', '#국제 이슈, #토론, #유엔', '#춤#스트릿댄스#공연', '#연극 #공연 #이무대의주인공은당신', '#사회문제 #사회과학학회 #연대활동 ', '#기독교동아리, #복음주의, #누구나함께', '# 알찬봉사 # 다양한 프로그램', '#유대감, #신앙, #여유', "#일본군'위안부' #평화 #인권", '#사랑이넘치는 #가족 #흘러가는사랑 #열방을품다', '#봉사, #친목, #자율참여', '# 영화 # 이야기', '#사회분과#봉사#KUSA', '#순수_창작_뮤지컬, #대학의_로망, #좋아서_하는_개고생', '#기독교, #말씀짝, #성경스케치', '#친목 #운동 #활기참', '#탁구 #sports #상시모집', '#고풋, #풋살, #futsal', '#친목 #경험 #문화', '#퀴어, #성소수자, #인권', '#학문 #철학 #사색', '#문학 #예술 #S급동방']
    index_list= random.sample(range(82),10)


    c_name =[]
    c_tag = []
    for i in range(10):
        c_name.append(club_name[index_list[i]])
        c_tag.append(club_f[index_list[i]])



    if request.method == "POST" :
        userid = request.POST.get("userid", '')
        rating = []
        for i in range(1,11) :
            rating.append(int(request.POST.get("rating" +str(i), ' ')))

        dataframe = pd.DataFrame({"userid": [userid] * 10,
                                      "club": c_name, "ratings": rating})

        r_result = new_recom(dataframe)

        return render(request, 'circlelens/reward_result.html', {'r_result':r_result, "c_name":c_name , "c_tag":c_tag})

    else :
        return render(request, "circlelens/reward.html", {"c_name":c_name , "c_tag":c_tag})






def survey(request) :
    if request.method == "POST" :
        receive = []
        s_name = ["학업", "소요시간", "친밀도", "동아리방_시설", "술문화", "기대하는_활동", "신입부원", "활동비", "친목행사", "동아리방_사용","수평적인_분위기", "소속감", "코로나19_방안", "활동적인_부원들", "학교외_활동", "초심자를_위한_활동", "동아리_장비", "소모임&스터디", "원하는활동_직접진행", "외부지원", "학과구성", "행사규모", "활동기간" ,"회비", "학년","활동강제", "동아리원_실력", "활동빈도", "활동장소", "활동주제","성비", "선발절차", "분야"]
        for i in range(33):
            sv = request.POST.get(s_name[i], "")
            receive.append(sv)

        result = classification(receive)
        return render(request, "circlelens/survey_result.html", { "result": result })

    else :
        return render(request, "circlelens/survey.html")

def login(request) :
    return render(request, "circlelens/login.html")

def keyword(request) :
    club = pd.read_csv("circlelens/contents.csv", encoding='CP949')
    if request.method == "POST" :
        sub = request.POST.get("sub","")
        main = request.POST.get("main","")
        result = club[club[sub] == 1]["동아리"].tolist()


        return render(request, "circlelens/keyword_result.html", {"result": result, "main": main, "sub": sub})
    else :
        return render(request, "circlelens/keyword.html")

def search(request) :
    intro = pd.read_csv('circlelens/club_introduce_DB.csv', encoding='cp949')
    club_list = pd.read_csv('circlelens/clubname.csv', encoding='cp949')
    intro.columns = ["동아리명",'공식사이트', '해시태그','소속' ,'동아리소개']
    if request.method == "POST" :
        club_name = request.POST.get("club", "")

        c_name = club_name

        site = intro[intro['동아리명'] == club_name]['공식사이트'].to_string()

        tag = intro[intro['동아리명'] == club_name]['해시태그'].to_string()

        where = intro[intro['동아리명'] == club_name]['소속'].to_string()

        introduce = intro[intro['동아리명'] == club_name]['동아리소개'].to_string()
        recom =[]
        recom = contents_based(c_name)+ item_based(c_name)
        return render(request, "circlelens/search_result.html", {"c_name": c_name, "site":site[2:], "tag": tag[2:], "where":where[2:], "introduce": introduce[2:], "recom":recom})
    else :
        return render(request, "circlelens/search.html")

def survey_result(request) :
    return render(request, "circlelens/survey_result.html")

def keyword_result(request) :
    return render(request, "circlelens/keyword_result.html")

def reward_result(request) :
    return render(request, "circlelens/reward_result.html")


    keyword_data = pd.read_csv('circlelens/contents.csv',encoding='cp949', index_col=0 )
    reward_data = pd.read_csv('circlelens/ratings.csv',encoding='cp949', index_col=0)
    intro = pd.read_csv('circlelens/club_introduce_data.csv',encoding='cp949', index_col=0)
    survey_data = pd.read_csv('circlelens/data.csv',encoding='cp949', index_col=0)
    intro = pd.read_csv('circlelens/matching.csv',encoding='cp949', index_col=0)

def search_result(request):
    return render(request, "circlelens/search_result.html")


