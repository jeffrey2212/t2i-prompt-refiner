def process_item(item):
    """Process a single item from the Civitai API response"""
    if not item:
        return None

    try:
        # Extract basic information
        meta = item.get("meta") or {}  # Use empty dict if meta is None
        model = meta.get("Model", item.get("baseModel", ""))  # Try baseModel as fallback
        prompt = meta.get("prompt", "")
        negative_prompt = meta.get("negativePrompt", "")

        # Get image information
        image_url = item.get("url", "")
        width = item.get("width", 0)
        height = item.get("height", 0)
        nsfw = item.get("nsfw", False)
        nsfw_level = item.get("nsfwLevel", "None")
        post_id = item.get("postId", "")
        username = item.get("username", "")

        # Get reactions and stats
        stats = item.get("stats", {})
        reaction_count = (
            stats.get("heartCount", 0) +
            stats.get("likeCount", 0) +
            stats.get("laughCount", 0) +
            stats.get("cryCount", 0)
        )
        comment_count = stats.get("commentCount", 0)

        # Create processed item
        processed_item = {
            "id": item.get("id", ""),
            "name": item.get("name", f"Image by {username}"),  # Use username if name not available
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_url": image_url,
            "width": width,
            "height": height,
            "nsfw": nsfw,
            "nsfw_level": nsfw_level,
            "post_id": post_id,
            "username": username,
            "reaction_count": reaction_count,
            "comment_count": comment_count,
            "created_at": item.get("createdAt", ""),
            "hash": item.get("hash", "")
        }

        # Validate required fields
        if not processed_item["id"] or not processed_item["image_url"]:
            print(f"Skipping item due to missing required fields: {processed_item}")
            return None

        return processed_item

    except Exception as e:
        print(f"Error processing item: {e}")
        print(f"Raw item data: {item}")
        return None
