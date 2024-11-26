import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()

search_url = "https://www.youtube.com/results?search_query=sinyaller+ve+sistemler&sp=EgIQAQ%253D%253D"
driver.get(search_url)
time.sleep(3)

try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "a.yt-simple-endpoint.inline-block.style-scope.ytd-thumbnail"))
    )
except:
    print("Öğeler bulunamadı veya sayfa yavaş yüklendi.")

video_elements = driver.find_elements(By.CSS_SELECTOR, "a.yt-simple-endpoint.inline-block.style-scope.ytd-thumbnail")
video_links = []
for video in video_elements:
    href = video.get_attribute("href")
    if href and "/shorts/" not in href:
        video_links.append(href)
    if len(video_links) == 10:
        break

print("Toplanan Video Linkleri:")
for link in video_links:
    print(link)

# CSV dosyasını başlat ve başlıkları ekle
with open("yorumlar.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Video Başlığı", "Yorum Yapan", "Yorum Tarihi", "Yorum", "Beğeni Sayısı", "Yanıt Sayısı"])

    seen_comments = set()

    for link in video_links:
        driver.get(link)
        time.sleep(5)

        try:
            video_title = driver.find_element(By.CSS_SELECTOR,
                                              "yt-formatted-string.style-scope.ytd-watch-metadata").text
        except Exception as e:
            print("Video başlığı bulunamadı:", e)
            video_title = "Başlık bulunamadı"

        time.sleep(2)
        try:
            comments_section = driver.find_element(By.CSS_SELECTOR, "ytd-item-section-renderer #contents")
            driver.execute_script("arguments[0].scrollIntoView();", comments_section)
            time.sleep(2)

            comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
            time.sleep(2)

            if not comment_elements:
                print(f"{video_title} için yorum bulunamadı.")
                continue

            last_position = None

            while True:
                for comment in comment_elements:
                    try:
                        user_name = comment.find_element(By.CSS_SELECTOR, "#author-text span").text.strip()
                    except:
                        user_name = "Kullanıcı adı bulunamadı"

                    try:
                        comment_text = comment.find_element(By.CSS_SELECTOR, "#content-text").text
                    except:
                        comment_text = "Yorum metni bulunamadı"

                    comment_unique_key = comment_text.strip()
                    if comment_unique_key in seen_comments:
                        continue


                    seen_comments.add(comment_unique_key)

                    try:
                        comment_date = comment.find_element(By.CSS_SELECTOR, "#published-time-text").text
                    except:
                        comment_date = "Tarih bulunamadı"

                    try:
                        likes = comment.find_element(By.CSS_SELECTOR, "#vote-count-middle").text
                        likes = likes if likes else "0"
                    except:
                        likes = "Beğeni sayısı bulunamadı"

                    try:
                        replies = comment.find_element(By.CSS_SELECTOR, "#replies #more-replies").text
                        reply_count = ''.join(filter(str.isdigit, replies))
                        reply_count = reply_count if reply_count else "0"
                    except:
                        reply_count = "0"

                    writer.writerow([video_title, user_name, comment_date, comment_text, likes, reply_count])

                    print(f"Yorum Yapan: {user_name}")
                    print(f"Yorum Tarihi: {comment_date}")
                    print(f"Yorum: {comment_text}")
                    print(f"Beğeni Sayısı: {likes}")
                    print(f"Yanıt Sayısı: {reply_count}")
                    print("=" * 50)


                if comment_elements:
                    last_comment = comment_elements[-1]
                    driver.execute_script("arguments[0].scrollIntoView();", last_comment)
                    time.sleep(3)

                    current_position = driver.execute_script("return window.pageYOffset;")
                    if current_position == last_position:
                        print("Tüm yorumlar yüklendi.")
                        break
                    last_position = current_position

                comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
                time.sleep(2)

            print(f"{video_title} için yorumlar alındı ve kaydedildi.\n")

        except Exception as e:
            print("Yorumlar alınırken bir hata oluştu:", e)

driver.quit()
