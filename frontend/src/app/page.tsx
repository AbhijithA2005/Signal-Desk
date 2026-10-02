import ChatInterface from "./chat-interface";
import { CrtIntro } from "./crt-intro";

export default function Home() {
	return (
		<CrtIntro>
			<ChatInterface />
		</CrtIntro>
	);
}
