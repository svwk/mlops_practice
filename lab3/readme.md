## 3 задание
*Автор: Савоськина С.В.*

Для создания образа и запуска контейнера в каталоге проекта выполнить команду
`docker-compose -f "docker-compose.yml" up -d --build `

Образ можно скачать
`docker pull svsav/lab3`
или отсюда
https://hub.docker.com/r/svsav/lab3

Запуск веб-приложения (после запуска котейнера):
http://localhost:8000/docs -> check_quality

Значения для проверки:
length: 230.33, width: 120.05 -> годная
length: 230.61, width: 120.10 -> брак
