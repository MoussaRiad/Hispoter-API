from fastapi import APIRouter, Depends
import imageRouter,modelRouter,userRouter

def get_data():
    return {"data": "Some data"}


main_router = APIRouter()

@main_router.get("/")
def index():
    return {"message": "Hello, FastAPI!"}

main_router.include_router(imageRouter.router)
main_router.include_router(userRouter.router)
main_router.include_router(modelRouter.router)