

from typing import List
from chatup_chat.core import Manager
from chatup_chat.core.admin.admin import Admin
from chatup_chat.core.cache import RedisClusterJson
from chatup_chat.core.exceptions import AdminFoundError
cache = RedisClusterJson()


class AdminManager(Manager):
    def __init__(self, room_manager: Manager = None):
        self.room_manager = room_manager

    def init_admin(self, shop_id: str, session_id: str) -> Admin:
        try:
            admin = self.get_admin(shop_id)
            admin.session_id = session_id
        except AdminFoundError:
            admin = Admin(session_id=session_id, shop_id=shop_id)
        self.remove_admin(admin)
        self.set_admin(admin)
        return admin

    def get_admin(self, shop_id: str) -> Admin:
        admins = cache.get_by_patterns(f"admin_{shop_id}")
        admins: List[Admin] = [Admin(**admin) for admin in admins]
        if len(admins) > 1:
            raise AdminFoundError("More than one admin found")
        elif len(admins) == 0:
            raise AdminFoundError("Admin not found")
        return admins[0]

    def get_admin_by_session(self, session_id: str) -> Admin:
        admins = cache.get_by_patterns(f"admin_*{session_id}")
        admins: List[Admin] = [Admin(**admin) for admin in admins]
        if len(admins) > 1:
            raise AdminFoundError("More than one admin found")
        elif len(admins) == 0:
            raise AdminFoundError("Admin not found")
        return admins[0]

    def set_admin(self, admin: Admin):
        cache[f"admin_{admin.shop_id}_{admin.session_id}"] = admin.to_dict()

    def get_space_admin(self, space_id) -> Admin:
        admins = cache.get_by_patterns(f"admin_{space_id}_*")
        if len(admins) > 1:
            raise AdminFoundError("More than one admin found")
        elif len(admins) == 0:
            raise AdminFoundError("Admin not found")
        return Admin(**admins[0])

    def checkout_admin(self, admin: Admin):
        cache.clear_cache(f"admin_*_{admin.session_id}")
        rooms = self.room_manager.get_live_rooms(admin.shop_id)
        for room in rooms:
            if room.admin_managed:
                room.admin_forfeited()

    def reset_connections(self):
        admins = self.get_space_admin("admin_*")
        for admin in admins:
            admin.notify_admin_of_off_room()

    def remove_admin(self, admin: Admin):
        cache.clear_cache(f"admin_{admin.shop_id}")
